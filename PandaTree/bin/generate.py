#!/usr/bin/env python

import sys
import os
from argparse import ArgumentParser
packdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(packdir + '/lib')
from panda import *

argParser = ArgumentParser(description = 'Generate C++ code for a flat tree')
argParser.add_argument('configs', metavar = 'CONFIG', nargs = '*', help = 'Tree definition files.')
argParser.add_argument('--clear-custom', '-C', dest = 'clear_custom', action = 'store_true', help = 'Clear custom code.')

args = argParser.parse_args()

if len(args.configs) == 0:
    def_files = []
    for fname in os.listdir(packdir + '/defs'):
        if fname.endswith('.def'):
            args.configs.append(packdir + '/defs/' + fname)

if args.clear_custom:
    common.PRESERVE_CUSTOM = False

# sort args.configs by requirement
cnames = []
requirements = {}
for fname in args.configs:
    if not fname.endswith('.def'):
        raise RuntimeError('Definition file must have file name ending with .def')

    cname = os.path.basename(fname)[:-4]

    cnames.append(cname)
    requirements[cname] = []

    configFile = open(fname)
    # <require> lines must be at the beginning of the file
    for line in configFile:
        if not line.startswith('<require '):
            break

        requirements[cname].append(line.strip().replace('<require ', '').replace('>', ''))

# iteratively sort
sorted_cnames = list(cnames)
while True:
    swapped = False
    for cname, req in requirements.items():
        imax = 0
        for pre in req:
            if pre not in cnames:
                raise RuntimeError('Required definition file ' + pre + ' not included')
    
            if cname in requirements[pre]:
                raise RuntimeError('Circular requirement: ' + cname + ' -> ' + pre + ' -> ' + cname)

            ip = sorted_cnames.index(pre)
            ic = sorted_cnames.index(cname)
            if ip > ic:
                sorted_cnames[ip], sorted_cnames[ic] = sorted_cnames[ic], sorted_cnames[ip]
                swapped = True

    if not swapped:
        break

fnames = list(args.configs)
args.configs = []
for cname in sorted_cnames:
    for fname in fnames:
        if os.path.basename(fname) == cname + '.def':
            args.configs.append(fname)
            break

# now we have a sorted list of configs

# globals
includes = [
    Include('#include "TTree.h"'),
    Include('#include "TString.h"'),
    Include('#include "Rtypes.h"')
]
typedefs = []
constants = []
asserts = []
enums = []
functions = []
phobjects = []
trees = []

# parse all config files
for fname in args.configs:
    configFile = open(fname)

    while True:
        line = configFile.readline()
        if line == '':
            break

        line = line.strip()
        
        if line == '':
            continue

        if line.startswith('<require '):
            continue

        if line.startswith('%'):
            #comment line
            continue

        try:
            includes.append(Include(line))
            continue
        except Definition.NoMatch:
            pass

        try:
            typedefs.append(Typedef(line))
            continue
        except Definition.NoMatch:
            pass

        try:
            constants.append(Constant(line, configFile))
            continue
        except Definition.NoMatch:
            pass

        try:
            asserts.append(Assert(line))
            continue
        except Definition.NoMatch:
            pass

        try:
            enums.append(Enum(line, configFile))
            continue
        except Definition.NoMatch:
            pass

        try:
            functions.append(Function(line, configFile))
            continue
        except Definition.NoMatch:
            pass

        try:
            phobjects.append(PhysicsObject(line, configFile))
            continue
        except Definition.NoMatch:
            pass

        try:
            trees.append(Tree(line, configFile))
            continue
        except Definition.NoMatch:
            pass

        print 'Skipping unrecognized pattern:', line

    configFile.close()

# create directories if necessary
if not os.path.isdir(common.PACKDIR + '/Objects/interface'):
    os.makedirs(common.PACKDIR + '/Objects/interface')
if not os.path.isdir(common.PACKDIR + '/Objects/src'):
    os.makedirs(common.PACKDIR + '/Objects/src')
if not os.path.isdir(common.PACKDIR + '/obj'):
    os.makedirs(common.PACKDIR + '/obj')

# are we running in a CMSSW environment?
if os.path.exists(common.PACKDIR + '/../../.SCRAM/Environment'):
    with open(common.PACKDIR + '/../../.SCRAM/Environment') as environment:
        if 'CMSSW' in environment.readline():
            # if so, we need to write the build file
            with open(common.PACKDIR + '/Objects/BuildFile.xml', 'w') as buildFile:
                buildFile.write('<use name="root"/>\n')
                buildFile.write('<use name="{PACKAGE}/Framework"/>\n'.format(PACKAGE = common.PACKAGE))
                buildFile.write('<export>\n')
                buildFile.write('  <lib name="1"/>\n')
                buildFile.write('</export>\n')

# write the globals file
header = FileOutput(common.PACKDIR + '/Objects/interface/Constants.h')
header.writeline('#ifndef {PACKAGE}_Objects_Constants_h'.format(PACKAGE = common.PACKAGE))
header.writeline('#define {PACKAGE}_Objects_Constants_h'.format(PACKAGE = common.PACKAGE))
header.newline()

for include in includes:
    include.write(header)

header.writeline('#include <cstring>')

header.newline()
header.writeline('namespace {NAMESPACE} {{'.format(NAMESPACE = common.NAMESPACE))
header.newline()
header.indent += 1

if len(typedefs) != 0:
    for typedef in typedefs:
        typedef.write(header)

    header.newline()

for enum in enums:
    enum.write_decl(header, context = 'global')
    header.newline()

if len(constants) != 0:
    for constant in constants:
        constant.write_decl(header, context = 'global')

    header.newline()

Assert.write(asserts, header)

for function in functions:
    function.write_decl(header, context = 'global')
    header.newline()

header.write_custom_block('Constants.h.global')
header.newline()

header.indent -= 1
header.writeline('}')

header.newline()

header.writeline('#endif')
header.close()

# .cc for constants
src = FileOutput(common.PACKDIR + '/Objects/src/Constants.cc')
src.writeline('#include "../interface/Constants.h"')

for enum in enums:
    src.newline()
    enum.write_def(src)

for function in functions:
    src.newline()
    function.write_def(src, context = 'global')

src.newline()
src.write_custom_block('Constants.cc.global')

src.close()

# write code for all objects and trees
for objdef in phobjects + trees:
    objdef.generate_header()
    objdef.generate_source()

# write a linkdef file
linkdef = FileOutput(common.PACKDIR + '/Objects/LinkDef.h')
for objdef in phobjects:
    linkdef.writeline('#include "PandaTree/Objects/interface/{name}.h"'.format(name = objdef.name))
for tree in trees:
    linkdef.writeline('#include "PandaTree/Objects/interface/{name}.h"'.format(name = tree.name))

linkdef.newline()

linkdef.writeline('#ifdef __CLING__')
linkdef.writeline('#pragma link off all globals;')
linkdef.writeline('#pragma link off all classes;')
linkdef.writeline('#pragma link off all functions;')

linkdef.writeline('#pragma link C++ nestedclass;')
linkdef.writeline('#pragma link C++ nestedtypedef;')
linkdef.writeline('#pragma link C++ namespace {NAMESPACE};'.format(NAMESPACE = common.NAMESPACE))

linkdef.newline()

for enum in enums:
    linkdef.writeline('#pragma link C++ enum {NAMESPACE}::{name};'.format(NAMESPACE = common.NAMESPACE, name = enum.name))

for objdef in phobjects:
    linkdef.writeline('#pragma link C++ class {NAMESPACE}::{name};'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))
    for branch in objdef.branches:
        if type(branch) is GenericBranch:
            linkdef.writeline('#pragma link C++ class {type}+;'.format(type = branch.type))
            linkdef.writeline('#pragma link C++ class std::vector<{type}>+;'.format(type = branch.type))

for objdef in phobjects:
    if not objdef.is_singlet():
        linkdef.writeline('#pragma link C++ class {NAMESPACE}::Array<{NAMESPACE}::{name}>;'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))
        linkdef.writeline('#pragma link C++ class {NAMESPACE}::Collection<{NAMESPACE}::{name}>;'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))
        linkdef.writeline('#pragma link C++ class {NAMESPACE}::Ref<{NAMESPACE}::{name}>;'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))
        linkdef.writeline('#pragma link C++ class {NAMESPACE}::RefVector<{NAMESPACE}::{name}>;'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))

for objdef in phobjects:
    if not objdef.is_singlet():
        linkdef.writeline('#pragma link C++ typedef {NAMESPACE}::{name}Array;'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))
        linkdef.writeline('#pragma link C++ typedef {NAMESPACE}::{name}Collection;'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))
        linkdef.writeline('#pragma link C++ typedef {NAMESPACE}::{name}Ref;'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))
        linkdef.writeline('#pragma link C++ typedef {NAMESPACE}::{name}RefVector;'.format(NAMESPACE = common.NAMESPACE, name = objdef.name))

for tree in trees:
    linkdef.writeline('#pragma link C++ class {NAMESPACE}::{name};'.format(NAMESPACE = common.NAMESPACE, name = tree.name))

linkdef.newline()

linkdef.writeline('#endif')
