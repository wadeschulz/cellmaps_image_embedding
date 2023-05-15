#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_image_embedding` package."""


import os
import shutil
import unittest
import tempfile
from cellmaps_image_embedding.runner import CellmapsImageEmbedder
from cellmaps_image_embedding.exceptions import CellMapsImageEmbeddingError


class TestCellmapsImageEmbeddingRunner(unittest.TestCase):
    """Tests for `cellmaps_image_embedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_constructor(self):
        """Tests constructor"""
        myobj = CellmapsImageEmbedder(outdir='foo')
        self.assertIsNotNone(myobj)

    def test_constructor_outdir_must_be_set(self):

        try:
            CellmapsImageEmbedder()
            self.fail('Expected exception')
        except CellMapsImageEmbeddingError as e:
            self.assertEqual('outdir is None', str(e))

    def test_run_image_dir_must_be_set(self):
        temp_dir = tempfile.mkdtemp()
        try:
            rundir = os.path.join(temp_dir, 'run')
            myobj = CellmapsImageEmbedder(outdir=rundir)
            myobj.run()
            self.fail('Expected exception')
        except CellMapsImageEmbeddingError as e:
            self.assertEqual('inputdir must be set', str(e))
        finally:
            shutil.rmtree(temp_dir)
