=======
History
=======

0.3.3 (2025-07-03)
--------------------

* Fix for UD-3116, missing header 0 row

0.3.2 (2025-05-13)
--------------------

* Updated to PEP 517 compliant build system

0.3.1 (2025-03-18)
-------------------

* Add version bounds for required packages.

* Bug fix for change in pytorch 2.6.

0.3.0 (2024-12-02)
-------------------

* Added README generation.

* Refactor code.

0.2.1 (2024-09-06)
-------------------

* Bug fix in ``--inputdir`` argument.

0.2.0 (2024-08-29)
-------------------

* Added ``--provenance`` flag to pass a path to json file with provenance information. This removes the
  necessity of input directory to be an RO-Crate.

* Bug fixes
    * Resolved an issue in embedding generation process where images associated with multiple genes were not correctly
      handled (ambiguous antibodies).

0.1.0 (2024-02-01)
------------------

* First release on PyPI.
