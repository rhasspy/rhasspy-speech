# OpenGrm NGram

This is a C++ library for making and modifying n-gram language models encoded as
weighted finite-state transducers. It uses OpenFst finite-state transducers
(FSTs) and FST archives (FARs) as inputs and outputs.

This library is primarily developed by [Brian Roark](mailto:roark@google.com).

## Dependencies

This library depends on:

*   A standards-compliant C++17 compiler (GCC \>= 7 or Clang \>= 700)
*   The most recent version of [OpenFst](http://openfst.org) built with the
    `grm` extensions (i.e., built with `./configure --enable-grm`) and headers

## Installation instructions

This library uses GNU autotools so one can simply use the standard `./configure;
make; make install` conventions for compilation and installation.

Alternatively, one can use [Bazel](https://bazel.build) to compile the libraries
by running `bazel build //:all` anywhere in the source tree.

## Usage

*   Binaries will normally be installed in `/usr/local/bin`
*   Libraries will normally be installed in `/usr/local/lib`
*   Headers will normally be installed in `/usr/local/include/ngram`

Linking is, by default, dynamic so that the `Fst` and `Arc` type DSO extensions
can be used correctly if desired.

## Documentation

See the [online documentation](https://ngram.opengrm.org) for more
documentation.

## License

This library is released under the Apache license. See [`LICENSE`](LICENSE) for
more information.

## Interested in contributing?

See [`CONTRIBUTING`](CONTRIBUTING) for more information.

## Mandatory disclaimer

This is not an official Google product.
