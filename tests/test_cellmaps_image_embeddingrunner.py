#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `cellmaps_image_embedding` package."""

import os
import shutil
import unittest
import tempfile
import csv
from unittest.mock import MagicMock, patch

from cellmaps_utils import constants
from cellmaps_utils.provenance import ProvenanceUtil
from cellmaps_image_embedding.runner import CellmapsImageEmbedder
from cellmaps_image_embedding.runner import FakeEmbeddingGenerator
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

    def test_run_with_fake_embedder(self):
        temp_dir = tempfile.mkdtemp()
        try:
            inputdir = os.path.join(temp_dir, 'inputdir')
            outdir = os.path.join(temp_dir, 'outdir')
            os.makedirs(inputdir, mode=0o755)

            img_gene_file = os.path.join(inputdir,
                                         '1_' +
                                         constants.IMAGE_GENE_NODE_ATTR_FILE)
            with open(img_gene_file, 'w') as f:
                f.write('name\trepresents\tambiguous\tantibody\tfilename\timageurl\n')
                f.write(
                    'PPFIBP1\tensembl:ENSG00000110841\t\tHPA001924\t35_H1_1_\thttp://images.proteinatlas.org/1924'
                    '/35_H1_1_blue_red_green.jpg\n')
                f.write(
                    'ACTN1\tensembl:ENSG00000072110\t\tCAB004303\t669_H5_1_\thttp://images.proteinatlas.org/4303'
                    '/669_H5_1_blue_red_green.jpg\n')
                f.write(
                    'MYO1B\tensembl:ENSG00000128641\t\tHPA013607\t107_F3_1_\thttp://images.proteinatlas.org/13607'
                    '/107_F3_1_blue_red_green.jpg\n')

            red_img_dir = os.path.join(inputdir, constants.RED)
            os.makedirs(red_img_dir, mode=0o755)
            for fake_img_file in ['35_H1_1_red.jpg', '669_H5_1_red.jpg',
                                  '107_F3_1_red.jpg']:
                open(os.path.join(red_img_dir, fake_img_file), 'a').close()

            prov_util = ProvenanceUtil()
            prov_util.register_rocrate(inputdir,
                                       name='name of input crate',
                                       organization_name='name of organization',
                                       project_name='name of project',
                                       description='description of rocrate',
                                       keywords=[''])
            gen = FakeEmbeddingGenerator(inputdir,
                                         dimensions=1024)
            x = CellmapsImageEmbedder(outdir=outdir,
                                      inputdir=inputdir,
                                      embedding_generator=gen,
                                      name='name of output crate',
                                      project_name='name of output proj',
                                      organization_name='name of output org',
                                      provenance_utils=prov_util,
                                      input_data_dict={'foo': 'value'})
            self.assertEqual(0, x.run())

            self.assertTrue(os.path.isfile(x.get_image_embedding_file()))
            genes = set()
            with open(x.get_image_embedding_file(), 'r') as f:
                reader = csv.reader(f, delimiter='\t')

                for row in reader:
                    if row[0] == '':
                        self.assertEqual(1024, len(row))
                    else:
                        self.assertEqual(1025, len(row))
                    genes.add(row[0])
            self.assertEqual({'', 'PPFIBP1', 'ACTN1', 'MYO1B'}, genes)

        finally:
            shutil.rmtree(temp_dir)

    def test_run_without_logging(self):
        """ Tests run() without logging."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(temp_dir, 'run')
            mock_embedding_generator = MagicMock()
            myobj = CellmapsImageEmbedder(outdir=run_dir,
                                          embedding_generator=mock_embedding_generator)
            try:
                myobj.run()
                self.fail('Expected CellMapsProvenanceError')
            except CellMapsImageEmbeddingError as e:
                print(e)
                self.assertTrue('inputdir' in str(e))

            self.assertFalse(os.path.isfile(os.path.join(run_dir, 'output.log')))
            self.assertFalse(os.path.isfile(os.path.join(run_dir, 'error.log')))

        finally:
            shutil.rmtree(temp_dir)

    def test_run_with_logging(self):
        """ Tests run() with logging."""
        temp_dir = tempfile.mkdtemp()
        try:
            run_dir = os.path.join(temp_dir, 'run')
            mock_embedding_generator = MagicMock()
            myobj = CellmapsImageEmbedder(outdir=run_dir,
                                          embedding_generator=mock_embedding_generator,
                                          skip_logging=False)
            try:
                myobj.run()
                self.fail('Expected CellMapsProvenanceError')
            except CellMapsImageEmbeddingError as e:
                self.assertTrue('inputdir' in str(e))

            self.assertTrue(os.path.isfile(os.path.join(run_dir, 'output.log')))
            self.assertTrue(os.path.isfile(os.path.join(run_dir, 'error.log')))

        finally:
            shutil.rmtree(temp_dir)


class TestFakeEmbeddingGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dimensions = 1024
        self.fake_img_dir = os.path.join(self.temp_dir, 'red')
        os.makedirs(self.fake_img_dir, exist_ok=True)

        self.fake_image_file = os.path.join(self.fake_img_dir, 'image_1_red.jpg')
        with open(self.fake_image_file, 'w') as f:
            f.write('fake content')

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('cellmaps_image_embedding.runner.ImageEmbeddingFilterAndNameTranslator._gen_filtered_mapping')
    def test_get_next_embedding(self, mock_gen_filtered_mapping):
        self.attributes_file = os.path.join(self.temp_dir, '1_image_gene_node_attributes.tsv')
        with open(self.attributes_file, 'w') as f:
            f.write('filename\tname\n')
            f.write('image_1_red.jpg\tGene1\n')
        mock_gen_filtered_mapping.return_value = {'image_1_': ['Gene1']}
        generator = FakeEmbeddingGenerator(self.temp_dir, self.dimensions)
        embeddings = list(generator.get_next_embedding())

        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0][0][0], 'Gene1')
        self.assertEqual(len(embeddings[0][0]), self.dimensions + 1)

    @patch('cellmaps_image_embedding.runner.ImageEmbeddingFilterAndNameTranslator._gen_filtered_mapping')
    def test_get_next_embedding_multiple_genes(self, mock_gen_filtered_mapping):
        self.attributes_file = os.path.join(self.temp_dir, '1_image_gene_node_attributes.tsv')
        with open(self.attributes_file, 'w') as f:
            f.write('filename\tname\n')
            f.write('image_1_red.jpg\tGene1\n')
            f.write('image_1_red.jpg\tGene2\n')
            f.write('image_1_red.jpg\tGene3\n')
        mock_gen_filtered_mapping.return_value = {'image_1_': ['Gene1', 'Gene2', 'Gene3']}
        generator = FakeEmbeddingGenerator(self.temp_dir, self.dimensions)
        embeddings = list(generator.get_next_embedding())

        self.assertEqual(len(embeddings), 3)
        for embedding, prob in embeddings:
            self.assertIn(embedding[0], ['Gene1', 'Gene2', 'Gene3'])
            self.assertEqual(len(embedding), self.dimensions + 1)
