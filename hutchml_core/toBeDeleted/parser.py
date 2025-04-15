#!/usr/bin/env python
"""
parser.py

This module provides a comprehensive parser for the entire MXQL grammar defined in mxql.lark.
It leverages the Lark parser and our MXQLTransformer (from transformer.py) to:
  - Parse any valid MXQL script.
  - Transform the Lark parse tree into a uniform AST represented as Python dictionaries.
  
This parser forms the first step in our flow, feeding the AST into our transpiler (transpiler.py),
which then produces PyCaret-compatible Python code.
"""

import os
from lark import Lark
from transformer import MXQLTransformer

class MXQLParser:
    def __init__(self, grammar_file='mxql.lark'):
        if not os.path.exists(grammar_file):
            raise FileNotFoundError(f"Grammar file '{grammar_file}' not found.")
        with open(grammar_file, 'r') as f:
            grammar = f.read()
        # Note: The grammar remains unchanged because "IN" is already used for experiment names.
        self.parser = Lark(grammar, start='start', parser='lalr', propagate_positions=True)
        self.transformer = MXQLTransformer()

    def parse_script(self, script):
        """
        Parse the provided MXQL script and return the Lark parse tree.
        """
        return self.parser.parse(script)

    def transform_script(self, script):
        """
        Parse and transform an MXQL script into an AST.
        Returns:
            A Python list containing one or more statement dictionaries.
        """
        tree = self.parse_script(script)
        ast = self.transformer.transform(tree)
        return ast

    def parse_and_print(self, script):
        """
        Parse the provided script and print a pretty-formatted parse tree.
        Useful for debugging and verifying that the grammar is covering all scenarios.
        """
        tree = self.parse_script(script)
        print(tree.pretty())
