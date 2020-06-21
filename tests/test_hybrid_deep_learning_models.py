#!/usr/bin/env python

"""Tests for `hybrid_deep_learning_models` package."""


import unittest
from click.testing import CliRunner

from hybrid_deep_learning_models import hybrid_deep_learning_models
from hybrid_deep_learning_models import cli


class TestHybrid_deep_learning_models(unittest.TestCase):
    """Tests for `hybrid_deep_learning_models` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'hybrid_deep_learning_models.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
