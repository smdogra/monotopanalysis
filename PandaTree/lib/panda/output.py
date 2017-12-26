import re
from common import *

class FileOutput(object):
    """
    Helper tool to write C++ code output.
    """

    def __init__(self, fname):
        self.custom_blocks = {}

        if PRESERVE_CUSTOM:
            try:
                original = open(fname)
                while True:
                    line = original.readline()
                    if not line:
                        break
                  
                    matches = re.match('/\* BEGIN CUSTOM (.+) \*/', line.strip())
                    if matches is None:
                        continue
    
                    block = []
                    while True:
                        line = original.readline()
                        if not line or '/* END CUSTOM */' in line:
                            break

                        block.append(line[:-1]) # remove the trailing newline character
    
                    self.custom_blocks[matches.group(1)] = block
                
                original.close()
            except IOError:
                pass

        self._file = open(fname, 'w')
        self.indent = 0

    def close(self):
        self._file.close()

    def write(self, text):
        self._file.write(text)

    def newline(self):
        self._file.write('\n')

    def writeline(self, line):
        self._file.write(('  ' * self.indent) + line + '\n')

    def writelines(self, lines, line_end = ''):
        indented_lines = []
        for line in lines:
            indented_lines.append(('  ' * self.indent) + line)

        self._file.write((line_end + '\n').join(indented_lines) + '\n')

    def write_custom_block(self, block_name, default = ''):
        self.writeline('/* BEGIN CUSTOM ' + block_name + ' */')

        if block_name not in self.custom_blocks or len(self.custom_blocks[block_name]) == 0:
            if default:
                self.writeline(default)
        else:
            self.write('\n'.join(self.custom_blocks[block_name]) + '\n')

        self.writeline('/* END CUSTOM */')
