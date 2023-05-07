#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_image_embedding` package."""


import os
import shutil
import unittest
import tempfile
from cellmaps_image_embedding.runner import CellmapsImageEmbeddingRunner
from cellmaps_image_embedding.exceptions import CellMapsImageEmbeddingError


class TestCellmapsImageEmbeddingRunner(unittest.TestCase):
    """Tests for `cellmaps_image_embedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_constructor(self):
        """Tests constructor"""
        myobj = CellmapsImageEmbeddingRunner()

        self.assertIsNotNone(myobj)

    def test_run_outdir_must_be_set(self):
        """ Tests run()"""
        myobj = CellmapsImageEmbeddingRunner()
        try:
            myobj.run()
            self.fail('Expected exception')
        except CellMapsImageEmbeddingError as e:
            self.assertEqual('outdir must be set', str(e))

    def test_run_image_dir_must_be_set(self):
        temp_dir = tempfile.mkdtemp()
        try:
            myobj = CellmapsImageEmbeddingRunner(outdir=temp_dir)
            myobj.run()
            self.fail('Expected exception')
        except CellMapsImageEmbeddingError as e:
            self.assertEqual('imagedir must be set', str(e))
        finally:
            shutil.rmtree(temp_dir)
