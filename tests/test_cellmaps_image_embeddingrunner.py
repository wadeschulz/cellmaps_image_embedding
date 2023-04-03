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

    def test_run_image_gene_node_attributes_must_be_set(self):
        temp_dir = tempfile.mkdtemp()
        try:
            myobj = CellmapsImageEmbeddingRunner(outdir=temp_dir,
                                                 imagedir=temp_dir)
            myobj.run()
            self.fail('Expected exception')
        except CellMapsImageEmbeddingError as e:
            self.assertEqual('image_gene_node_attributes '
                             'must be set', str(e))
        finally:
            shutil.rmtree(temp_dir)

    def test_run_success(self):
        temp_dir = tempfile.mkdtemp()
        try:
            subdir = os.path.join(temp_dir, 'subdir')
            attrfile = os.path.join(temp_dir, 'attr.tsv')
            with open(attrfile, 'w') as f:
                f.write('a\tb\nc\t\d\n')

            myobj = CellmapsImageEmbeddingRunner(outdir=subdir,
                                                 imagedir=temp_dir,
                                                 image_gene_node_attributes=attrfile)
            self.assertEqual(0, myobj.run())
            self.assertTrue(os.path.isdir(subdir))
            image_emd = os.path.join(subdir, 'image_emd.tsv')
            self.assertTrue(os.path.isfile(image_emd))


        finally:
            shutil.rmtree(temp_dir)
