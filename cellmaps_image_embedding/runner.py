#! /usr/bin/env python

import os
import sys
import time
from datetime import date
import logging
import shutil
import csv
import random
import warnings
import requests
import torch
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from cellmaps_utils import constants
import cellmaps_image_embedding
from cellmaps_utils import logutils
from cellmaps_utils.provenance import ProvenanceUtil
from cellmaps_image_embedding.exceptions import CellMapsImageEmbeddingError
from cellmaps_image_embedding.dataset import *
from cellmaps_image_embedding.models import *

logger = logging.getLogger(__name__)

ABB_LABEL_INDEX = {
    "0": "Nucleoplasm",
    "1": "N. membrane",
    "2": "Nucleoli",
    "3": "N. fibrillar c.",
    "4": "N. speckles",
    "5": "N. bodies",
    "6": "ER",
    "7": "Golgi app.",
    "8": "Peroxisomes",
    "9": "Endosomes",
    "10": "Lysosomes",
    "11": "Int. fil.",
    "12": "Actin fil.",
    "13": "F. a. sites",
    "14": "Microtubules",
    "15": "M. ends",
    "16": "Cyt. bridge",
    "17": "Mitotic spindle",
    "18": "MTOC",
    "19": "Centrosome",
    "20": "Lipid droplets",
    "21": "PM",
    "22": "C. Junctions",
    "23": "Mitochondria",
    "24": "Aggresome",
    "25": "Cytosol",
    "26": "C. bodies",
    "27": "Rods & Rings"
}


class EmbeddingGenerator(object):
    """
    Base class for implementations that generate
    network embeddings
    """
    DEFAULT_FOLD = 1
    DIMENSIONS = 1024
    SUFFIX = '.jpg'

    def __init__(self, dimensions=DIMENSIONS, fold=DEFAULT_FOLD):
        """
        Constructor
        """
        self._dimensions = dimensions
        self._fold = fold
        self._fairscape_dataset_tuples = []

    def get_dimensions(self):
        """
        Gets number of dimensions this embedding will generate

        :return: number of dimensions aka vector length
        :rtype: int
        """
        return self._dimensions

    def get_fold(self):
        """
        Gets fold
        :return:
        :rtype: int
        """
        return self._fold

    def get_next_embedding(self):
        """
        Generator method for getting next embedding.
        Caller should implement with ``yield`` operator

        :raises: NotImplementedError: Subclasses should implement this
        :return: Embedding
        :rtype: list
        """
        raise NotImplementedError('Subclasses should implement')

    def get_datasets_that_need_to_be_registered(self):
        """
        Gets any datasets that need to be registered with FAIRSCAPE

        :return: list of tuples in format of (dict, filepath as str)
        :rtype: list
        """
        return self._fairscape_dataset_tuples


class FakeEmbeddingGenerator(EmbeddingGenerator):
    """
    Fakes image embedding
    """

    def __init__(self, inputdir, dimensions=EmbeddingGenerator.DIMENSIONS, fold=EmbeddingGenerator.DEFAULT_FOLD,
                 suffix=EmbeddingGenerator.SUFFIX, img_emd_translator=None):
        """
        Constructor

        :param inputdir: Directory where images reside under
                         red, green, blue, and yellow directories
        :type inputdir: str
        :param dimensions: Desired size of output embedding
        :type dimensions: int
        :param suffix: Image suffix with starting ``.``
        :type suffix: str
        """
        super().__init__(dimensions=dimensions, fold=fold)
        self._inputdir = inputdir
        self._suffix = suffix
        if img_emd_translator is None:
            self._img_emd_translator = ImageEmbeddingFilterAndNameTranslator(image_downloaddir=inputdir, fold=fold)
        warnings.warn(constants.IMAGE_EMBEDDING_FILE +
                      ' contains FAKE DATA!!!!\n'
                      'You have been warned\nHave a nice day\n')
        logger.error(constants.IMAGE_EMBEDDING_FILE +
                     ' contains FAKE DATA!!!! '
                     'You have been warned. Have a nice day')

    def _get_image_id_list(self):
        """
        Looks at red directory under image directory to
        get a list of image ids which are the file names
        in that directory with last ``_`` and everything to
        the right of it removed from the file name
        :return:
        """
        image_set = set()
        red_image_dir = os.path.join(self._inputdir, constants.RED)
        for entry in os.listdir(red_image_dir):
            if not entry.endswith(self._suffix):
                continue
            if not os.path.isfile(os.path.join(red_image_dir, entry)):
                continue
            # include the _ at the end cause that is also included in
            # image_gene_node_attributes.tsv file
            image_set.add(entry[: entry.rfind('_') + 1])
        return list(image_set)

    def get_next_embedding(self):
        """
        Generator method for getting next embedding.
        Caller should implement with ``yield`` operator

        :raises: NotImplementedError: Subclasses should implement this
        :return: Embedding
        :rtype: list
        """
        for image_id in self._get_image_id_list():
            if image_id not in self._img_emd_translator.get_name_mapping():
                continue
            genes = self._img_emd_translator.get_name_mapping()[image_id]
            for g in genes:
                row = [g]
                row.extend(np.random.normal(size=self.get_dimensions()))  # sample normal distribution
                prob = [g]
                prob.extend(
                    [random.random() for x in range(0, len(ABB_LABEL_INDEX.keys()))])  # might need to add to one
                yield row, prob


class DensenetEmbeddingGenerator(EmbeddingGenerator):
    """
    Runs densenet bundled with this tool via command line to
    generate embedding. Why do it this way? Easier transition
    from the original
    `Densenet <https://github.com/CellProfiling/densenet>`__
    code and no memory leaks

    """

    def __init__(self, inputdir, dimensions=EmbeddingGenerator.DIMENSIONS,
                 outdir=None,
                 model_path=None,
                 suffix=EmbeddingGenerator.SUFFIX,
                 fold=EmbeddingGenerator.DEFAULT_FOLD,
                 img_emd_translator=None):
        """
        Constructor

        :param inputdir: Directory where red, blue, green, and yellow
                         image directories reside
        :type inputdir: str
        :param dimensions: Desired size of output embedding vector
        :type dimensions: int
        :param pythonbinary: Path to python binary, if set to ``None``
                             the version of python that invoked this
                             command will be used
        :type pythonbinary: str
        :param predict: Path to prediction script. Default value is the
                        script bundled with this tool
        :type predict: str
        :param model_path: Path to model file
        :type model_path: str
        :param suffix: Image suffix with starting ``.``
        :type suffix: str
        :param img_emd_translator:
        """
        super().__init__(dimensions=dimensions, fold=fold)
        self._outdir = outdir
        self._inputdir = inputdir
        self._gpus = ''
        self._image_size = 1536
        self._crop_size = 1024
        self._device = 'cpu'
        self._cuda_available = False
        self._model_path = model_path
        self._suffix = suffix
        self._channels = 4
        self._num_classes = 28
        self._seeds = [0]
        self._augments = ['default']
        self._model = None
        self._dataset = None
        self._dataloader = None

        if img_emd_translator is None:
            self._img_emd_translator = ImageEmbeddingFilterAndNameTranslator(image_downloaddir=inputdir,
                                                                             fold=fold)

    def _initialize_model(self):
        """

        """
        model = class_densenet121_large_dropout(num_classes=self._num_classes,
                                                in_channels=self._channels,
                                                pretrained=self._model_path)
        model = DataParallel(model)

        # TODO: Need to see if this is necessary
        #       Need to properly support cpu and gpu modes
        model.to(self._device)

        #
        # If a node has a GPU you get an error
        # description of fix
        # https://stackoverflow.com/questions/68551032/is-there-a-way-to-use-torch-nn-dataparallel-with-cpu
        # and line below is the fix.
        model = model.module.to(self._device)

        model = model.eval()
        return model

    def _initialize_dataset(self):
        """

        :return:
        """
        dataset = ProteinDataset(
            self._inputdir,
            self._outdir,
            image_size=self._image_size,
            crop_size=self._crop_size,
            in_channels=self._channels,
            suffix=self._suffix,
            alt_image_ids=None)
        return dataset

    def _initialize_dataloader(self):
        """

        :return:
        """
        dataloader = DataLoader(
            self._dataset,
            sampler=SequentialSampler(self._dataset),
            batch_size=1,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )
        return dataloader

    def _download_model(self, model_path):
        """
        If model_path is a URL attempt to download it
        to pipeline directory, otherwise return as is

        :param model_path: URL or file path to model file needed
                           for image embedding
        :type model_path: str
        :return: path to model file
        :rtype: str
        """
        dest_file = os.path.abspath(os.path.join(self._outdir, 'model.pth'))
        self._update_fairscape_dataset_tuples(dest_model=dest_file,
                                              src_url=model_path)
        if os.path.isfile(model_path):
            shutil.copy(model_path, dest_file)
            return dest_file

        with requests.get(model_path,
                          stream=True) as r:
            content_size = int(r.headers.get('content-length', 0))
            tqdm_bar = tqdm(desc='Downloading ' + os.path.basename(model_path),
                            total=content_size,
                            unit='B', unit_scale=True,
                            unit_divisor=1024)
            logger.debug('Downloading ' + str(model_path) +
                         ' of size ' + str(content_size) +
                         'b to ' + dest_file)
            try:
                r.raise_for_status()
                with open(dest_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        tqdm_bar.update(len(chunk))
            finally:
                tqdm_bar.close()
        return dest_file

    def _update_fairscape_dataset_tuples(self, dest_model=None, src_url=None):
        """
        Registers model.pth file with create as a dataset

        """
        data_dict = {'name': 'Densenet model file',
                     'description': 'Trained Densenet model used for classification of IF images'
                                    ' from ' + str(src_url),
                     'data-format': 'pth',
                     'keywords': ['Trained Densenet model', 'pytorch', 'classification'],
                     'author': cellmaps_image_embedding.__name__,
                     'version': cellmaps_image_embedding.__version__,
                     'date-published': date.today().strftime('%Y-%m-%d')}
        self._fairscape_dataset_tuples.append((data_dict, dest_model))

    def get_next_embedding(self):
        """
        Generator method for getting next embedding.

        :return: Embedding vector with 1st element
        :rtype: list
        """
        self._model_path = self._download_model(self._model_path)
        self._model = self._initialize_model()
        self._dataset = self._initialize_dataset()
        self._dataloader = self._initialize_dataloader()

        for seed in self._seeds:
            for augment in self._augments:
                np.random.seed(seed)
                torch.manual_seed(seed)
                # eg. augment_default
                transform = eval("augment_%s" % augment)
                self._dataloader.dataset.set_transform(transform=transform)
                random_crop = (self._crop_size > 0) and (seed != 0)
                self._dataloader.dataset.set_random_crop(random_crop=random_crop)
                image_ids = np.array(self._dataloader.dataset.image_ids)

                for iter_index, (images, indices) in tqdm(
                    enumerate(self._dataloader, 0), total=len(self._dataloader)
                ):
                    with torch.no_grad():
                        if self._cuda_available:
                            images = Variable(images.cuda())
                        else:
                            images = Variable(images)
                        logits, features = self._model(images)

                        image_id = image_ids[iter_index] + '_'
                        if image_id not in self._img_emd_translator.get_name_mapping():
                            continue
                        genes = self._img_emd_translator.get_name_mapping()[image_id]
                        probs = F.sigmoid(logits).cpu().data.numpy().tolist()[0]
                        features = features.cpu().data.numpy().tolist()

                        for g in genes:
                            # probabilities
                            prob_list = [g]
                            prob_list.extend(probs)

                            # features
                            row = [g]
                            row.extend(features[0])
                            yield row, prob_list

    def get_datasets_that_need_to_be_registered(self):
        """
        Gets model.pth dataset that needs to be registered with FAIRSCAPE.

        .. warning::

            Must not be called before invocation of :meth:`~cellmaps_image_embedding.runner.DensenetEmbeddingGenerator.get_next_embedding`

        :raises CellMapsImageEmbeddingError: If this method is called before at least one
                                             invocation of :meth:`~cellmaps_image_embedding.runner.DensenetEmbeddingGenerator.get_next_embedding`
        :return: list of tuples in format of (dict, filepath as str)
        :rtype: list
        """
        if len(self._fairscape_dataset_tuples) == 0:
            raise CellMapsImageEmbeddingError('get_next_embedding must be called at least '
                                              'once before invoking this method')
        return self._fairscape_dataset_tuples


class ImageEmbeddingFilterAndNameTranslator(object):
    """
    Converts image embedding names and filters keeping only
    one per gene

    """

    def __init__(self, image_downloaddir=None, fold=1):
        """
        Constructor
        """
        self._id_to_gene_mapping = self._gen_filtered_mapping(os.path.join(image_downloaddir, str(fold) + '_' +
                                                                           constants.IMAGE_GENE_NODE_ATTR_FILE))

    def _gen_filtered_mapping(self, image_gene_node_attrs_file):
        """
        Reads TSV file

        :param image_gene_node_attrs_file:
        :return:
        """
        mapping_dict = {}
        with open(image_gene_node_attrs_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                f = row['filename'].split(',')[0]
                if f not in mapping_dict:
                    mapping_dict[f] = []
                mapping_dict[f].append(row['name'])
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Mapping dict: ' + str(mapping_dict))
        return mapping_dict

    def get_name_mapping(self):
        """
        Gets mapping of old name to new name

        :return: mapping of old name to new name
        :rtype: dict
        """
        return self._id_to_gene_mapping


class CellmapsImageEmbedder(object):
    """
    Class to run algorithm
    """

    def __init__(self, outdir=None,
                 inputdir=None,
                 embedding_generator=None,
                 skip_logging=True,
                 name=None,
                 organization_name=None,
                 project_name=None,
                 input_data_dict=None,
                 provenance_utils=ProvenanceUtil(),
                 provenance=None):
        """
        Constructor

        :param outdir: Directory to write the results of this tool
        :type outdir: str
        :param inputdir: Directory with images that will be embedded, containing color-separated subdirectorie.
                         Output directory from cellmaps_imagedownloader.
        :type inputdir: str
        :param embedding_generator: An object implementing the embedding generation logic,
                                    typically a subclass of :py:class:`~EmbeddingGenerator`, such as
                                    :py:class:`~DensenetEmbeddingGenerator` or :py:class:`~FakeEmbeddingGenerator`.
        :type embedding_generator: EmbeddingGenerator
        :param skip_logging: If ``True`` skip logging, if ``None`` or ``False`` do NOT skip logging
        :type skip_logging: bool
        :param name: Optional name for the dataset, used in provenance tracking. If not provided,
                     it will be inferred from the RO-Crate in the input directory or fallback defaults.
        :type name: str or None
        :param organization_name: Optional name of the organization creating the dataset.
                                  Used for provenance metadata.
        :type organization_name: str or None
        :param project_name: Optional name of the project associated with the dataset.
                             Used for provenance metadata.
        :type project_name: str or None
        :param input_data_dict: Dictionary of input parameters used for execution. This is captured
                                and stored in metadata as part of reproducibility and provenance.
                                If not provided, a default dictionary is generated.

                                Example:

                                .. code-block:: python

                                    {'outdir': '/output/path', 'inputdir': '/input/path'}
        :type input_data_dict: dict or None
        :param provenance_utils: Utility class instance for handling RO-Crate creation,
                                 software registration, and FAIRSCAPE dataset tracking.
                                 Default is a new :py:class:`~ProvenanceUtil` object.
        :type provenance_utils: ProvenanceUtil
        :param provenance: Optional dictionary containing explicit provenance metadata, such as
                           dataset name, project, organization, keywords, and description.
                           If present, it will override missing information in RO-Crate.

                           Example:

                           .. code-block:: python

                               {
                                   'name': 'Cell Image Embedding Dataset',
                                   'organization-name': 'CM4AI',
                                   'project-name': 'Cell Image Project',
                                   'keywords': ['embedding', 'microscopy'],
                                   'description': 'Embedding of IF microscopy images.'
                               }
        :type provenance: dict or None
        """
        logger.debug('In constructor')
        if outdir is None:
            raise CellMapsImageEmbeddingError('outdir is None')
        self._outdir = os.path.abspath(outdir)
        self._inputdir = os.path.abspath(inputdir) if inputdir is not None else inputdir
        self._start_time = int(time.time())
        self._name = name
        self._project_name = project_name
        self._organization_name = organization_name
        self._provenance_utils = provenance_utils
        self._embedding_generator = embedding_generator
        self._softwareid = None
        self._input_data_dict = input_data_dict
        self._generated_dataset_ids = []
        self._keywords = None
        self._description = None
        self._provenance = provenance
        self._inputdataset_ids = []
        if skip_logging is None:
            self._skip_logging = False
        else:
            self._skip_logging = skip_logging

        if self._input_data_dict is None:
            self._input_data_dict = {'outdir': self._outdir,
                                     'inputdir': self._inputdir,
                                     'embedding_generator': str(self._embedding_generator),
                                     'name': self._name,
                                     'project_name': self._project_name,
                                     'organization_name': self._organization_name,
                                     'skip_logging': self._skip_logging,
                                     'provenance': str(self._provenance)
                                     }

    def _update_provenance_fields(self):
        """

        :return:
        """
        if os.path.exists(os.path.join(self._inputdir, constants.RO_CRATE_METADATA_FILE)):
            prov_attrs = self._provenance_utils.get_merged_rocrate_provenance_attrs(self._inputdir,
                                                                                    override_name=self._name,
                                                                                    override_project_name=self._project_name,
                                                                                    override_organization_name=self._organization_name,
                                                                                    extra_keywords=[
                                                                                        'IF Image Embedding',
                                                                                        'IF microscopy images',
                                                                                        'embedding',
                                                                                        'fold' +
                                                                                        str(self._embedding_generator.get_fold())])
            if self._name is None:
                self._name = prov_attrs.get_name()

            if self._organization_name is None:
                self._organization_name = prov_attrs.get_organization_name()

            if self._project_name is None:
                self._project_name = prov_attrs.get_project_name()
            self._keywords = prov_attrs.get_keywords()
            self._description = prov_attrs.get_description()
        elif self._provenance is not None:
            self._name = self._provenance['name'] if 'name' in self._provenance else 'Image Embedding'
            self._organization_name = self._provenance['organization-name'] \
                if 'organization-name' in self._provenance else 'NA'
            self._project_name = self._provenance['project-name'] \
                if 'project-name' in self._provenance else 'NA'
            self._keywords = self._provenance['keywords'] if 'keywords' in self._provenance else ['image']
            self._description = self._provenance['description'] if 'description' in self._provenance else \
                'Embedding of Images'
        else:
            raise CellMapsImageEmbeddingError('Input directory should be an RO-Crate or provenance should be '
                                              'specified.')

    def _create_rocrate(self):
        """
        Creates rocrate for output directory

        :raises CellMapsProvenanceError: If there is an error
        """
        try:
            self._provenance_utils.register_rocrate(self._outdir,
                                                    name=self._name,
                                                    organization_name=self._organization_name,
                                                    project_name=self._project_name,
                                                    description=self._description,
                                                    keywords=self._keywords)
        except TypeError as te:
            raise CellMapsImageEmbeddingError('Invalid provenance: ' + str(te))
        except KeyError as ke:
            raise CellMapsImageEmbeddingError('Key missing in provenance: ' + str(ke))

    def _create_output_directory(self):
        """
        Creates output directory if it does not already exist

        :raises CellmapsDownloaderError: If output directory is None or if directory already exists
        """
        if os.path.isdir(self._outdir):
            raise CellMapsImageEmbeddingError(self._outdir + ' already exists')

        os.makedirs(self._outdir, mode=0o755)
        for cur_color in constants.COLORS:
            cdir = os.path.join(self._outdir, cur_color + '_resize')
            if not os.path.isdir(cdir):
                logger.debug('Creating directory: ' + cdir)
                os.makedirs(cdir,
                            mode=0o755)

    def _register_software(self):
        """
        Registers this tool

        :raises CellMapsImageEmbeddingError: If fairscape call fails
        """
        software_keywords = self._keywords
        software_keywords.extend(['tools', cellmaps_image_embedding.__name__])
        software_description = self._description + ' ' + \
                               cellmaps_image_embedding.__description__
        self._softwareid = self._provenance_utils.register_software(self._outdir,
                                                                    name=cellmaps_image_embedding.__name__,
                                                                    description=software_description,
                                                                    author=cellmaps_image_embedding.__author__,
                                                                    version=cellmaps_image_embedding.__version__,
                                                                    file_format='py',
                                                                    keywords=software_keywords,
                                                                    url=cellmaps_image_embedding.__repo_url__)

    def _register_computation(self):
        """
        Registers computation with FAIRSCAPE

        """
        logger.debug('Getting id of input rocrate')
        if os.path.exists(os.path.join(self._inputdir, constants.RO_CRATE_METADATA_FILE)):
            self._inputdataset_ids.append(self._provenance_utils.get_id_of_rocrate(self._inputdir))

        keywords = self._keywords
        keywords.extend(['computation'])
        description = self._description + ' run of ' + cellmaps_image_embedding.__name__

        self._provenance_utils.register_computation(self._outdir,
                                                    name=cellmaps_image_embedding.__computation_name__,
                                                    run_by=str(self._provenance_utils.get_login()),
                                                    command=str(self._input_data_dict),
                                                    description=description,
                                                    keywords=keywords,
                                                    used_software=[self._softwareid],
                                                    used_dataset=self._inputdataset_ids,
                                                    generated=self._generated_dataset_ids)

    def _register_image_embedding_file(self):
        """
        Registers :py:const:`cellmaps_utils.constants.IMAGE_EMBEDDING_FILE` file with
        create as a dataset

        """
        logger.debug('Registering embedding file with FAIRSCAPE')
        description = self._description
        description += ' file'
        keywords = self._keywords
        keywords.extend(['file'])

        data_dict = {'name': cellmaps_image_embedding.__name__ + ' output file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'tsv',
                     'author': cellmaps_image_embedding.__name__,
                     'version': cellmaps_image_embedding.__version__,
                     'schema': 'https://raw.githubusercontent.com/fairscape/cm4ai-schemas/main/v0.1.0/'
                               'cm4ai_schema_image_embedding_emd.json',
                     'date-published': date.today().strftime(self._provenance_utils.get_default_date_format_str())}
        dset_id = self._provenance_utils.register_dataset(self._outdir,
                                                          source_file=self.get_image_embedding_file(),
                                                          data_dict=data_dict,
                                                          skip_copy=True)
        self._generated_dataset_ids.append(dset_id)

    def _register_image_probability_file(self):
        """
        Registers :py:const:`cellmaps_utils.constants.IMAGE_LABELS_PROBABILITY_FILE` file with
        create as a dataset

        """
        logger.debug('Registering label probability file with FAIRSCAPE')
        description = self._description
        description += ' file'
        keywords = self._keywords
        keywords.extend(['file'])

        data_dict = {'name': cellmaps_image_embedding.__name__ + ' output file',
                     'description': description,
                     'keywords': keywords,
                     'data-format': 'tsv',
                     'author': cellmaps_image_embedding.__name__,
                     'version': cellmaps_image_embedding.__version__,
                     'schema': 'https://raw.githubusercontent.com/fairscape/cm4ai-schemas/main/v0.1.0/cm4ai_schema_image_embedding_labels_prob.json',
                     'date-published': date.today().strftime(self._provenance_utils.get_default_date_format_str())}
        dset_id = self._provenance_utils.register_dataset(self._outdir,
                                                          source_file=self.get_image_probability_file(),
                                                          data_dict=data_dict,
                                                          skip_copy=True)
        self._generated_dataset_ids.append(dset_id)

    def _register_embedding_generator_datasets(self):
        """

        :return:
        """
        for dset_tuple in self._embedding_generator.get_datasets_that_need_to_be_registered():
            dset_id = self._provenance_utils.register_dataset(self._outdir,
                                                              source_file=dset_tuple[1],
                                                              data_dict=dset_tuple[0],
                                                              skip_copy=True)
            logger.debug('Adding embedding_generator dataset to fairscape: ' + str(dset_tuple))
            self._generated_dataset_ids.append(dset_id)

    def get_image_embedding_file(self):
        """
        Gets image embedding file
        :return:
        """
        return os.path.join(self._outdir, constants.IMAGE_EMBEDDING_FILE)

    def get_image_probability_file(self):
        """
        Gets image probability file
        :return:
        """
        return os.path.join(self._outdir, constants.IMAGE_LABELS_PROBABILITY_FILE)

    def generate_readme(self):
        description = getattr(cellmaps_image_embedding, '__description__', 'No description provided.')
        version = getattr(cellmaps_image_embedding, '__version__', '0.0.0')

        with open(os.path.join(os.path.dirname(__file__), 'readme_outputs.txt'), 'r') as f:
            readme_outputs = f.read()

        readme = readme_outputs.format(DESCRIPTION=description, VERSION=version)
        with open(os.path.join(self._outdir, 'README.txt'), 'w') as f:
            f.write(readme)

    def _write_task_start_json(self):

        data = {'imagedir': self._inputdir}

        if self._input_data_dict is not None:
            data.update({'commandlineargs': self._input_data_dict})

        logutils.write_task_start_json(outdir=self._outdir,
                                       start_time=self._start_time,
                                       data=data,
                                       version=cellmaps_image_embedding.__version__)

    def run(self):
        """
        Runs cellmaps_image_embedding


        :return:
        """
        exitcode = 99
        try:
            logger.debug('In run method')
            self._create_output_directory()

            if self._skip_logging is False:
                logutils.setup_filelogger(outdir=self._outdir,
                                          handlerprefix='cellmaps_image_embedding')

            self._write_task_start_json()

            self.generate_readme()

            if self._inputdir is None:
                raise CellMapsImageEmbeddingError('inputdir must be set')

            self._update_provenance_fields()

            self._create_rocrate()

            self._register_software()

            # generate result
            raw_embeddings = []
            with open(self.get_image_embedding_file(), 'w', newline='') as f:
                with open(self.get_image_probability_file(), 'w', newline='') as pf:
                    writer = csv.writer(f, delimiter='\t')
                    prob_writer = csv.writer(pf, delimiter='\t')
                    header_line = ['id']
                    header_line.extend([x for x in range(self._embedding_generator.get_dimensions())])
                    writer.writerow(header_line)
                    header_line_prob = ['']
                    for key in range(0, len(ABB_LABEL_INDEX.keys())):
                        header_line_prob.append(ABB_LABEL_INDEX[str(key)])
                    prob_writer.writerow(header_line_prob)
                    for row, prob_list in self._embedding_generator.get_next_embedding():
                        writer.writerow(row)
                        raw_embeddings.append(row)
                        prob_writer.writerow(prob_list)

            self._register_image_embedding_file()
            self._register_image_probability_file()
            self._register_embedding_generator_datasets()

            self._register_computation()
            exitcode = 0
        finally:
            logutils.write_task_finish_json(outdir=self._outdir,
                                            start_time=self._start_time,
                                            status=exitcode)

        return exitcode
