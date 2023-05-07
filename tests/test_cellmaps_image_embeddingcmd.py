#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_image_embedding` package."""

import os
import tempfile
import shutil

import unittest
from cellmaps_image_embedding import cellmaps_image_embeddingcmd


class TestCellmapsImageEmbedding(unittest.TestCase):
    """Tests for `cellmaps_image_embedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_parse_arguments(self):
        """Tests parse arguments"""
        res = cellmaps_image_embeddingcmd._parse_arguments('hi', ['foo',
                                                                  '--inputdir', 'imagedir'])

        self.assertEqual(0, res.verbose)
        self.assertEqual('foo', res.outdir)
        self.assertEqual('imagedir', res.inputdir)
        self.assertEqual(res.logconf, None)

        someargs = ['-vv', '--logconf', 'hi',
                    'foo',
                    '--inputdir', 'imagedir']
        res = cellmaps_image_embeddingcmd._parse_arguments('hi', someargs)

        self.assertEqual(2, res.verbose)
        self.assertEqual('hi', res.logconf)

    def test_main(self):
        """Tests main function"""

        # try where loading config is successful
        try:
            temp_dir = tempfile.mkdtemp()
            res = cellmaps_image_embeddingcmd.main(['myprog.py',
                                                    'foo',
                                                    '--inputdir', 'imagedir'])
            self.assertEqual(res, 2)
        finally:
            shutil.rmtree(temp_dir)
