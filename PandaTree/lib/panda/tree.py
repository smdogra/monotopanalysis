from common import *
from base import Definition
from obj import Object
from output import FileOutput

class Tree(Definition, Object):
    """
    Tree definition. Definition file syntax:

    {Name(>Parent)}
    <branch definitions>
    <function definitions>
    """

    def __init__(self, line, source):
        Definition.__init__(self, line, '\\{([A-Z][a-zA-Z0-9]+)(>[A-Z][a-zA-Z0-9]+|)\\}$')
        Object.__init__(self, self.matches.group(1), source)

        if self.matches.group(2):
            self.parent = self.matches.group(2)[1:]
        else:
            self.parent = 'TreeEntry'

    def generate_header(self):
        """
        Write the header file.
        """

        header = FileOutput('{PACKDIR}/Objects/interface/{name}.h'.format(PACKDIR = PACKDIR, name = self.name))
        header.writeline('#ifndef {PACKAGE}_Objects_{name}_h'.format(PACKAGE = PACKAGE, name = self.name))
        header.writeline('#define {PACKAGE}_Objects_{name}_h'.format(PACKAGE = PACKAGE, name = self.name))
        if self.parent == 'TreeEntry':
            header.writeline('#include "../../Framework/interface/TreeEntry.h"')
        else:
            header.writeline('#include "{parent}.h"'.format(parent = self.parent))
        header.writeline('#include "Constants.h"')

        included = []
        for objbranch in self.objbranches:
            if objbranch.objname not in included:
                header.writeline('#include "{brobj}.h"'.format(brobj = objbranch.objname))
                included.append(objbranch.objname)

        for include in self.includes:
            include.write(header)

        header.newline()
        header.writeline('namespace {NAMESPACE} {{'.format(NAMESPACE = NAMESPACE))
        header.newline()
        header.indent += 1

        header.writeline('class {name} : public {parent} {{'.format(name = self.name, parent = self.parent))
        header.writeline('public:')
        header.indent += 1
        
        header.writeline('{name}();'.format(name = self.name)) # default constructor
        header.writeline('{name}({name} const&);'.format(name = self.name)) # copy constructor
        header.writeline('~{name}();'.format(name = self.name)) # destructor
        header.writeline('{name}& operator=({name} const&);'.format(name = self.name)) # assignment operator

        header.newline()

        header.writeline('void print(std::ostream& = std::cout, UInt_t level = 1) const override;')
        header.writeline('void dump(std::ostream& = std::cout) const override;')

        if len(self.functions) != 0:
            header.newline()
            for function in self.functions:
                function.write_decl(header, context = 'class')

        if len(self.objbranches) != 0:
            header.newline()
            for objbranch in self.objbranches:
                objbranch.write_decl(header)

        if len(self.branches) != 0:
            header.newline()
            for branch in self.branches:
                branch.write_decl(header, context = 'TreeEntry')

        header.newline()
        header.writeline('static utils::BranchList getListOfBranches(Bool_t direct = kFALSE);')

        header.newline()
        header.indent -= 1
        header.writeline('protected:')
        header.indent += 1
        header.writeline('void doSetStatus_(TTree&, utils::BranchList const&) override;')
        header.writeline('utils::BranchList doGetStatus_(TTree&) const override;')
        header.writeline('utils::BranchList doGetBranchNames_() const override;')
        header.writeline('void doSetAddress_(TTree&, utils::BranchList const&, Bool_t setStatus) override;')
        header.writeline('void doBook_(TTree&, utils::BranchList const&) override;')
        header.writeline('void doGetEntry_(TTree&) override;')
        header.writeline('void doInit_() override;')
        header.writeline('void doUnlink_(TTree&) override;')

        header.newline()
        header.indent -= 1
        header.writeline('public:')
        header.indent += 1
        header.write_custom_block('{name}.h.classdef'.format(name = self.name))

        header.indent -= 1

        header.writeline('};')

        header.newline()
        header.write_custom_block('{name}.h.global'.format(name = self.name))

        header.newline()

        header.indent -= 1
        header.writeline('}')
        header.newline()

        header.writeline('#endif')
        header.close()

    def generate_source(self):
        """
        Write the .cc file.
        """

        collections = [b for b in self.objbranches if b.conttype == 'Collection']

        src = FileOutput('{PACKDIR}/Objects/src/{name}.cc'.format(PACKDIR = PACKDIR, name = self.name))
        src.writeline('#include "../interface/{name}.h"'.format(name = self.name))
        src.newline()

        src.writeline('{NAMESPACE}::{name}::{name}() :'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.indent += 1
        src.writeline('{parent}()'.format(parent = self.parent, name = self.name))
        src.indent -= 1
        src.writeline('{')
        src.indent += 1
        if len(self.objbranches) != 0:
            src.writeline('std::vector<Object*> myObjects{{' + ', '.join(['&{name}'.format(name = b.name) for b in self.objbranches]) + '}};')
            src.writeline('objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());')

        if len(collections) != 0:
            src.writeline('std::vector<CollectionBase*> myCollections{{' + ', '.join(['&{name}'.format(name = b.name) for b in collections]) + '}};')
            src.writeline('collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());')

        if len(self.references) != 0:
            src.newline()
            for ref in self.references:
                ref.write_def(src, self.objbranches)

        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('{NAMESPACE}::{name}::{name}({name} const& _src) :'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.indent += 1
        initializers = ['{parent}(_src)'.format(parent = self.parent)]
        for objbranch in self.objbranches:
            initializers.append(objbranch.cpyctor())
        for branch in self.branches:
            branch.init_copy(initializers, context = 'TreeEntry')
        src.writelines(initializers, ',')
        src.indent -= 1
        src.writeline('{')
        src.indent += 1
        if len(self.objbranches) != 0:
            src.writeline('std::vector<Object*> myObjects{{' + ', '.join(['&{name}'.format(name = b.name) for b in self.objbranches]) + '}};')
            src.writeline('objects_.insert(objects_.end(), myObjects.begin(), myObjects.end());')

        if len(collections) != 0:
            src.writeline('std::vector<CollectionBase*> myCollections{{' + ', '.join(['&{name}'.format(name = b.name) for b in collections]) + '}};')
            src.writeline('collections_.insert(collections_.end(), myCollections.begin(), myCollections.end());')

        if len(self.branches) != 0:
            src.newline()
            for branch in self.branches:
                if branch.is_array():
                    branch.write_assign(src, context = 'TreeEntry')

        if len(self.references) != 0:
            src.newline()
            for ref in self.references:
                ref.write_def(src, self.objbranches)

        src.write_custom_block('{name}.cc.copy_ctor'.format(name = self.name))

        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('{NAMESPACE}::{name}::~{name}()'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1

        src.write_custom_block('{name}.cc.dtor'.format(name = self.name))

        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('{NAMESPACE}::{name}&'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{NAMESPACE}::{name}::operator=({name} const& _src)'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        src.writeline('{parent}::operator=(_src);'.format(parent = self.parent))

        src.newline()
        src.write_custom_block('{name}.cc.operator='.format(name = self.name))
        src.newline()

        if len(self.branches) != 0:
            for branch in self.branches:
                branch.write_assign(src, context = 'TreeEntry')
            src.newline()
        if len(self.objbranches) != 0:
            for objbranch in self.objbranches:
                objbranch.write_assign(src)
            src.newline()
        if len(self.references) != 0:
            for ref in self.references:
                ref.write_def(src, self.objbranches)
            src.newline()

        src.writeline('return *this;')
        src.indent -= 1
        src.writeline('}')

        src.newline()
        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        src.write_custom_block('{name}.cc.print'.format(name = self.name), default = 'dump(_out);')
        src.indent -= 1
        src.writeline('}')

        src.newline()
        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::dump(std::ostream& _out/* = std::cout*/) const'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1

        if self.parent != 'TreeEntry':
            src.writeline('{parent}::dump(_out);'.format(parent = self.parent))
            src.newline()

        if len(self.branches) != 0:
            for branch in self.branches:
                branch.write_dump(src)
            src.newline()
        if len(self.objbranches) != 0:
            for objbranch in self.objbranches:
                objbranch.write_dump(src)
            src.newline()
        
        src.indent -= 1
        src.writeline('}')

        src.writeline('/*static*/')
        src.writeline('panda::utils::BranchList')
        src.writeline('{NAMESPACE}::{name}::getListOfBranches(Bool_t _direct/* = kFALSE*/)'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        src.writeline('utils::BranchList blist;')
        if self.parent != 'TreeEntry':
            src.writeline('blist += {parent}::getListOfBranches(_direct);'.format(parent = self.parent))
            src.newline()

        src.writeline('blist += {{{bnames}}};'.format(bnames = ', '.join('"{name}"'.format(name = branch.name) for branch in self.branches if '!' not in branch.modifier)))
        if len(self.objbranches) != 0:
            src.writeline('if (!_direct) {')
            src.indent += 1

            for objbranch in self.objbranches:
                src.writeline('blist += {otype}::getListOfBranches().fullNames("{name}");'.format(otype = objbranch.objname, name = objbranch.name))

            src.indent -= 1
            src.writeline('}')

        src.writeline('return blist;')
        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('/*protected*/')
        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::doSetStatus_(TTree& _tree, utils::BranchList const& _branches)'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        if self.parent != 'TreeEntry':
            src.writeline('{parent}::doSetStatus_(_tree, _branches);'.format(parent = self.parent))
        for branch in self.branches:
            branch.write_set_status(src, context = 'TreeEntry')
        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('/*protected*/')
        src.writeline('panda::utils::BranchList')
        src.writeline('{NAMESPACE}::{name}::doGetStatus_(TTree& _tree) const'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        src.writeline('utils::BranchList blist;')
        if self.parent != 'TreeEntry':
            src.writeline('blist += {parent}::doGetStatus_(_tree);'.format(parent = self.parent))
        src.newline()
        for branch in self.branches:
            branch.write_get_status(src, context = 'TreeEntry')
        src.writeline('return blist;')
        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('/*protected*/')
        src.writeline('panda::utils::BranchList')
        src.writeline('{NAMESPACE}::{name}::doGetBranchNames_() const'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        src.writeline('return getListOfBranches(true);')
        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('/*protected*/')
        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::doSetAddress_(TTree& _tree, utils::BranchList const& _branches, Bool_t _setStatus)'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        if self.parent != 'TreeEntry':
            src.writeline('{parent}::doSetAddress_(_tree, _branches, _setStatus);'.format(parent = self.parent))
            src.newline()

        for branch in self.branches:
            branch.write_set_address(src, context = 'TreeEntry')
        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('/*protected*/')
        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::doBook_(TTree& _tree, utils::BranchList const& _branches)'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        if self.parent != 'TreeEntry':
            src.writeline('{parent}::doBook_(_tree, _branches);'.format(parent = self.parent))
            src.newline()

        for branch in self.branches:
            branch.write_book(src, context = 'TreeEntry')
        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('/*protected*/')
        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::doGetEntry_(TTree& _tree)'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        if self.parent != 'TreeEntry':
            src.writeline('{parent}::doGetEntry_(_tree);'.format(parent = self.parent))
            src.newline()

        src.write_custom_block('{name}.cc.doGetEntry_'.format(name = self.name))
        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::doInit_()'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        if self.parent != 'TreeEntry':
            src.writeline('{parent}::doInit_();'.format(parent = self.parent))
            src.newline()

        for branch in self.branches:
            branch.write_init(src, context = 'TreeEntry')

        src.write_custom_block('{name}.cc.doInit_'.format(name = self.name))
        src.indent -= 1
        src.writeline('}')
        src.newline()

        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::doUnlink_(TTree& _tree)'.format(NAMESPACE = NAMESPACE, name = self.name))
        src.writeline('{')
        src.indent += 1
        if self.parent != 'TreeEntry':
            src.writeline('{parent}::doUnlink_(_tree);'.format(parent = self.parent))
            src.newline()

        src.write_custom_block('{name}.cc.doUnlink_'.format(name = self.name))
        src.indent -= 1
        src.writeline('}')
        src.newline()

        if len(self.functions) != 0:
            src.newline()
            for function in self.functions:
                function.write_def(src, context = self.name)

        src.newline()
        src.write_custom_block('{name}.cc.global'.format(name = self.name))

        src.close()
