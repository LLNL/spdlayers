# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2022-07-13

### Added
- added the option for the Eigen layer to support semi-definite predictions
- new option `n_zero_eigvals` for Eigen layer to specify how many zero eigenvalues the model should expect

### Changed
- Eigen layer now uses `torch.linalg.eigh` instead of `torch.linalg.eig`

## [0.2.0] - 2022-06-23

### Changed
- add 'None' as a positivity function
- set the default Cholesky positivity function to 'None'
- add citations to methods used on README.md

## [0.1.1] - 2022-04-01

### Changed
- add our preprint to the readme

## [0.1.0] - 2022-03-24

### Changed
- `spdlayers.tools.in_shape_from` now should always output integers (as opposed to floats)

## [0.0.3] - 2022-02-06

### Changed
- `spdlayers.tools.in_shape_from` now uses the sum of the first n natural numbers, see https://cseweb.ucsd.edu/groups/tatami/handdemos/sum/

## [0.0.2] - 2021-12-02

### Added
- Add url homepage in setup.py to point to this github repo

### Changed
- tweak readme installation notes to just use pip to install

### Fixed
- bug in single source version, which prevented install of spdlayers if torch was not already installed
- mistake in the pypi license classifier

## [0.0.1] - 2021-11-19
### Added
- First public release!
