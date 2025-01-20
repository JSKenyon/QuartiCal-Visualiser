# QuartiCal-Visualiser

The QuartiCal-Visualiser is a convenient tool for visualising the gain solutions
produced by [QuartiCal](https://github.com/ratt-ru/QuartiCal). It also allows 
for interactive flagging of those solutions. 

## Installation

QuartiCal-Visualiser can be installed from PyPI by running
`pip install qcvisualiser`. Alternatively, developers can install it directly
from source using either Poetry or pip.

## Usage

QuartiCal-Visualiser can be run from the command line using
`govisualise path/to/gain`. You can then navigate to `localhost:5006` in
your browser to interact with the gains. 

## Options

At present, QuartiCal-Visualiser has limited options. All of them can be 
displayed by running `govisualise --help`.

## Remote Viewing

As QuartiCal starts a web server, it is possible to interact with it remotely.
This can be accomplished by port-forwarding e.g. 
`ssh -L 5006:localhost:5006 user@remote` before proceeding as detailed above.