# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Highlight printed TVM script.
"""

from pygments import highlight as phl
from pygments.lexers import Python3Lexer
from pygments.formatters import Terminal256Formatter
from pygments.style import Style
from pygments.token import Keyword, Name, Comment, String, Number, Operator


class VSCDark(Style):
    """A VSCode-Dark-like Pygment style configuration"""

    styles = {
        Keyword: "bold #c586c0",
        Keyword.Namespace: "#4ec9b0",
        Keyword.Type: "#82aaff",
        Name: "#9cdcfe",
        Name.Function: "bold #dcdcaa",
        Name.Class: "bold #569cd6",
        Name.Decorator: "italic #fe4ef3",
        String: "#ce9178",
        Number: "#b5cea8",
        Operator: "#bbbbbb",
        Operator.Word: "#569cd6",
        Comment: "italic #6a9956",
    }


def highlight(text: str) -> str:
    """
    Highlight given TVM script string with Pygment

    Parameters
    ----------
    text : str
        String of the TVM script

    Returns
    -------
    str
        The resulting string highlighted by ANSI escape code
    """
    return phl(text, Python3Lexer(), Terminal256Formatter(style=VSCDark))
