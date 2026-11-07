import os

import config


def test_dotenv_loader_reads_values_and_real_env_wins(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nCFG_TEST_A=hello\nCFG_TEST_B = 'quoted value'\nCFG_TEST_C=from_file\nnot a pair\n",
        encoding="utf-8",
    )
    os.environ["CFG_TEST_C"] = "from_environment"
    try:
        config._load_dotenv(env_file)
        assert os.environ["CFG_TEST_A"] == "hello"
        assert os.environ["CFG_TEST_B"] == "quoted value"
        assert os.environ["CFG_TEST_C"] == "from_environment"
    finally:
        for key in ("CFG_TEST_A", "CFG_TEST_B", "CFG_TEST_C"):
            os.environ.pop(key, None)


def test_missing_dotenv_is_ignored(tmp_path):
    config._load_dotenv(tmp_path / "does-not-exist.env")
