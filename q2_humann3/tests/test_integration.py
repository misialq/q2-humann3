# ----------------------------------------------------------------------------
# Copyright (c) 2026, Bokulich Lab.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import gzip
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import pandas as pd
from qiime2 import Artifact
from qiime2.plugin.testing import TestPluginBase
from q2_types.per_sample_sequences import (
    CasavaOneEightSingleLanePerSampleDirFmt,
)

from q2_humann3.plugin_setup import plugin
from q2_humann3._types_and_formats import (
    HumannDatabaseDirFmt,
    HumannReactionDirectoryFormat,
    MetaphlanDatabaseDirFmt,
)
from q2_sapienns.plugin_setup import (
    HumannGeneFamilyDirectoryFormat,
    HumannPathAbundanceDirectoryFormat,
    MetaphlanMergedAbundanceDirectoryFormat,
)


class RunHumannIntegrationTests(TestPluginBase):
    package = "q2_humann3.tests"

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)

    def _make_reads_artifact_for_fixtures(
        self, sample_fixtures: dict[str, str]
    ) -> Artifact:
        reads = CasavaOneEightSingleLanePerSampleDirFmt()
        for sample_id, fixture_name in sample_fixtures.items():
            source_path = Path(self.get_data_path(fixture_name))
            destination_path = reads.path / f"{sample_id}_S1_L001_R1_001.fastq.gz"
            with source_path.open("rb") as source_fh:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=destination_path.open("wb"),
                    mtime=0,
                ) as destination_fh:
                    destination_fh.write(source_fh.read())

        return Artifact.import_data("SampleData[SequencesWithQuality]", reads)

    def _make_humann_demo_database(
        self,
        package_dir: str,
        payload_dir: str,
        database_kind: str,
        build: str,
    ) -> HumannDatabaseDirFmt:
        try:
            import humann
        except ImportError:
            self.skipTest("HUMANN package is not importable.")

        source_dir = Path(humann.__file__).resolve().parent / "data" / package_dir
        if not source_dir.exists():
            self.skipTest(f"HUMANN demo database is missing: {source_dir}")

        database = HumannDatabaseDirFmt()
        shutil.copytree(source_dir, database.path / payload_dir)

        # Ensure at least one file with the required extension exists
        extension = ".ffn.gz" if database_kind == "chocophlan" else ".dmnd"
        if not list((database.path / payload_dir).glob(f"*{extension}")):
            (database.path / payload_dir / f"demo{extension}").write_bytes(b"data")

        with (database.path / "metadata.json").open("w") as fh:
            json.dump(
                {"database_kind": database_kind, "build": build},
                fh,
            )
        return database

    def _make_humann_demo_database_artifact(
        self,
        package_dir: str,
        payload_dir: str,
        database_kind: str,
        build: str,
        semantic_type: str,
    ) -> Artifact:
        return Artifact.import_data(
            semantic_type,
            self._make_humann_demo_database(
                package_dir, payload_dir, database_kind, build
            ),
        )

    def _make_toy_metaphlan_database(self) -> MetaphlanDatabaseDirFmt:
        database = MetaphlanDatabaseDirFmt()
        index = "mpa_vTest"
        (database.path / f"{index}.pkl").write_bytes(b"taxonomy")
        for suffix in ("1.bt2", "2.bt2", "3.bt2", "4.bt2", "rev.1.bt2", "rev.2.bt2"):
            (database.path / f"{index}.{suffix}").write_bytes(b"bowtie")

        with (database.path / "metadata.json").open("w") as fh:
            json.dump(
                {"database_kind": "metaphlan", "index": index},
                fh,
            )
        return database

    def _make_toy_metaphlan_database_artifact(self) -> Artifact:
        return Artifact.import_data(
            "MetaphlanDatabase", self._make_toy_metaphlan_database()
        )

    def _read_result_table(self, artifact, directory_format) -> pd.DataFrame:
        table_path = artifact.view(directory_format).path / "table.tsv"
        lines = table_path.read_text().splitlines()
        while lines and lines[0].startswith("#mpa_"):
            lines.pop(0)
        return pd.read_csv(StringIO("\n".join(lines)), sep="\t")

    def _assert_result_tables_equal(
        self,
        observed,
        expected,
        directory_format,
        key_columns: list[str] = None,
    ) -> None:
        observed_df = self._read_result_table(observed, directory_format)
        expected_df = self._read_result_table(expected, directory_format)
        if key_columns is None:
            key_columns = [expected_df.columns[0]]

        sample_columns = sorted(
            column for column in expected_df.columns if column not in key_columns
        )
        columns = key_columns + sample_columns
        observed_df = (
            observed_df[columns].sort_values(key_columns).reset_index(drop=True)
        )
        expected_df = (
            expected_df[columns].sort_values(key_columns).reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(observed_df, expected_df, check_dtype=False)

    def _write_fake_metaphlan_executable(self, bin_dir: Path) -> None:
        executable = bin_dir / "metaphlan"
        executable.write_text("""#!/usr/bin/env python
from pathlib import Path
import sys

if "--version" in sys.argv:
    print("MetaPhlAn version 3.0.0")
    raise SystemExit(0)

profile = \"\"\"#mpa_v30_CHOCOPhlAn_201901
#mpa_v30_CHOCOPhlAn_201901
#SampleID\tMetaphlan_Analysis
#clade_name\tNCBI_tax_id\trelative_abundance
k__Bacteria\t2\t100.0
g__Bacteroides|s__Bacteroides_dorei\t357276\t100.0
\"\"\"

Path(sys.argv[sys.argv.index("-o") + 1]).write_text(profile)
Path(sys.argv[sys.argv.index("--bowtie2out") + 1]).write_text("")
""")
        executable.chmod(0o755)

    def test_partitioned_pipeline_matches_unpartitioned_run(self):
        missing_executables = [
            executable
            for executable in (
                "humann",
                "humann_join_tables",
                "merge_metaphlan_tables.py",
                "humann_regroup_table",
                "bowtie2",
                "diamond",
            )
            if shutil.which(executable) is None
        ]
        if missing_executables:
            self.skipTest(
                "Missing integration-test executables: "
                + ", ".join(missing_executables)
            )

        reads = self._make_reads_artifact_for_fixtures(
            {
                "demo-a": "humann-demo.fastq",
                "demo-b": "humann-demo.fastq",
            }
        )
        nucleotide_database = self._make_humann_demo_database_artifact(
            "chocophlan_DEMO",
            "chocophlan",
            "chocophlan",
            "full",
            "HumannDatabase[ChocoPhlAn]",
        )
        translated_search_database = self._make_humann_demo_database_artifact(
            "uniref_DEMO",
            "uniref",
            "translated-search",
            "uniref90_diamond",
            "HumannDatabase[TranslatedSearch]",
        )
        metaphlan_database = self._make_toy_metaphlan_database_artifact()
        fake_bin_dir = self.temp_path / "bin"
        fake_bin_dir.mkdir()
        self._write_fake_metaphlan_executable(fake_bin_dir)
        run_humann = plugin.actions["run_humann"]
        run_humann_unpartitioned = plugin.actions["_run_humann"]

        with patch.dict(
            os.environ,
            {"PATH": f"{fake_bin_dir}{os.pathsep}{os.environ['PATH']}"},
        ):
            expected = run_humann_unpartitioned(
                reads=reads,
                nucleotide_database=nucleotide_database,
                translated_search_database=translated_search_database,
                metaphlan_database=metaphlan_database,
            )
            observed = run_humann(
                reads=reads,
                nucleotide_database=nucleotide_database,
                translated_search_database=translated_search_database,
                metaphlan_database=metaphlan_database,
                num_partitions=2,
            )

        self._assert_result_tables_equal(
            observed.gene_families,
            expected.gene_families,
            HumannGeneFamilyDirectoryFormat,
        )
        self._assert_result_tables_equal(
            observed.path_abundance,
            expected.path_abundance,
            HumannPathAbundanceDirectoryFormat,
        )
        self._assert_result_tables_equal(
            observed.metaphlan_profile,
            expected.metaphlan_profile,
            MetaphlanMergedAbundanceDirectoryFormat,
            key_columns=["clade_name", "NCBI_tax_id"],
        )
        self._assert_result_tables_equal(
            observed.reactions,
            expected.reactions,
            HumannReactionDirectoryFormat,
        )
