#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_image_embedding` package."""


import unittest
from cellmaps_image_embedding.runner import CellmapsImageEmbeddingRunner


class TestCellmapsimageembeddingrunner(unittest.TestCase):
    """Tests for `cellmaps_image_embedding` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_constructor(self):
        """Tests constructor"""
        myobj = CellmapsImageEmbeddingRunner()

        self.assertIsNotNone(myobj)

    def test_run(self):
        """ Tests run()"""
        myobj = CellmapsImageEmbeddingRunner()
        self.assertEqual(0, myobj.run())
