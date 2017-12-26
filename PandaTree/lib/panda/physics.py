from common import *
from base import Definition
from obj import Object
from branch import Branch
from output import FileOutput

class PhysicsObject(Definition, Object):
    """
    Physics object definition. Definition file syntax:
    
    [Name(>Parent)(:SINGLE)]
    <variable definitions>
    <function definitions>
    """

    _known_objects = []

    @staticmethod
    def get(name):
        try:
            return next(o for o in PhysicsObject._known_objects if o.name == name)
        except StopIteration:
            raise KeyError('PhysicsObject.get({name})'.format(name = name))

    def __init__(self, line, source):
        """
        Argument: re match object
        """

        Definition.__init__(self, line, '\\[([A-Z][a-zA-Z0-9]+)(>[A-Z][a-zA-Z0-9]+|)(\\:SINGLE|)\\] *$')
        PhysicsObject._known_objects.append(self)

        # is this a singlet?
        if self.matches.group(3) == ':SINGLE':
            self.parent = 'Singlet'
            self._singlet = True
        else:
            self.parent = 'Element'
            self._singlet = False

        # if >parent is present, update the parent class name
        if self.matches.group(2):
            self.parent = self.matches.group(2)[1:]
            self._singlet = None

        Object.__init__(self, self.matches.group(1), source)

        if len([f for f in self.functions if f.is_pure_virtual]) == 0:
            self.instantiable = True
        else:
            self.instantiable = False

    def is_singlet(self):
        if self._singlet is None:
            return PhysicsObject.get(self.parent).is_singlet()
        else:
            return self._singlet

    def generate_header(self):
        """
        Write a header file.
        """

        inheritance = [self]
        while True:
            try:
                inheritance.insert(0, PhysicsObject.get(inheritance[0].parent))
            except KeyError:
                break

        header = FileOutput('{PACKDIR}/Objects/interface/{name}.h'.format(PACKDIR = PACKDIR, name = self.name))
        header.writeline('#ifndef {PACKAGE}_Objects_{name}_h'.format(PACKAGE = PACKAGE, name = self.name))
        header.writeline('#define {PACKAGE}_Objects_{name}_h'.format(PACKAGE = PACKAGE, name = self.name))
        header.writeline('#include "Constants.h"')

        included = ['#include "{name}.h"'.format(name = self.name)]
        if self.parent == 'Element':
            header.writeline('#include "../../Framework/interface/Element.h"')
        elif self.parent == 'Singlet':
            header.writeline('#include "../../Framework/interface/Singlet.h"')
        else:
            stmt = '#include "{parent}.h"'.format(parent = self.parent)
            if stmt not in included:
                header.writeline(stmt)
                included.append(stmt)

        header.writeline('#include "../../Framework/interface/Array.h"')
        header.writeline('#include "../../Framework/interface/Collection.h"')
        header.writeline('#include "../../Framework/interface/Ref.h"')
        header.writeline('#include "../../Framework/interface/RefVector.h"')

        for include in self.includes:
            if include.code not in included:
                include.write(header)
                included.append(include.code)

        for branch in self.branches:
            if hasattr(branch, 'refname'): # is a RefBranch
                stmt = '#include "{obj}.h"'.format(obj = branch.objname)
                if stmt not in included:
                    header.writeline(stmt)
                    included.append(stmt)

        header.newline()
        header.writeline('namespace {NAMESPACE} {{'.format(NAMESPACE = NAMESPACE))
        header.newline()
        header.indent += 1

        header.writeline('class {name} : public {parent} {{'.format(name = self.name, parent = self.parent))
        header.writeline('public:')
        header.indent += 1

        if len(self.enums) != 0:
            for enum in self.enums:
                enum.write_decl(header, context = 'class')
            header.newline()

        if len(self.constants) != 0:
            for constant in self.constants:
                constant.write_decl(header, context = 'class')
            header.newline()
        
        if not self.is_singlet():
            header.writeline('struct datastore : public {parent}::datastore {{'.format(parent = self.parent))
            header.indent += 1

            header.writeline('datastore() : {parent}::datastore() {{}}'.format(parent = self.parent))
            header.writeline('~datastore() { deallocate(); }')

            newline = False
            for ancestor in inheritance:
                if len(ancestor.branches) == 0:
                    continue

                if not newline:
                    header.newline()
                    newline = True

                if ancestor != self:
                    header.writeline('/* {name}'.format(name = ancestor.name))

                for branch in ancestor.branches:
                    branch.write_decl(header, context = 'datastore')

                if ancestor != self:
                    header.writeline('*/')

            header.newline()
            header.writeline('void allocate(UInt_t n) override;')
            header.writeline('void deallocate() override;')
            header.writeline('void setStatus(TTree&, TString const&, utils::BranchList const&) override;')
            header.writeline('utils::BranchList getStatus(TTree&, TString const&) const override;')
            header.writeline('utils::BranchList getBranchNames(TString const& = "") const override;')
            header.writeline('void setAddress(TTree&, TString const&, utils::BranchList const& = {"*"}, Bool_t setStatus = kTRUE) override;')
            header.writeline('void book(TTree&, TString const&, utils::BranchList const& = {"*"}, Bool_t dynamic = kTRUE) override;')
            header.writeline('void releaseTree(TTree&, TString const&) override;')
            header.writeline('void resizeVectors_(UInt_t) override;')

            header.indent -= 1
            header.writeline('};')
            header.newline()

            header.writeline('typedef Array<{name}> array_type;'.format(name = self.name, parent = self.parent))
            header.writeline('typedef Collection<{name}> collection_type;'.format(name = self.name, parent = self.parent))
            header.newline()

        header.writeline('typedef {parent} base_type;'.format(parent = self.parent))
        header.newline()

        # "boilerplate" functions (default ctor for non-SINGLE objects are pretty nontrivial though)
        if self.instantiable:
            header.writeline('{name}(char const* name = "");'.format(name = self.name)) # default constructor
            header.writeline('{name}({name} const&);'.format(name = self.name)) # copy constructor

        # standard constructors and specific functions
        if not self.is_singlet():
            # whereas elements of collections are constructed from an array and an index
            header.writeline('{name}(datastore&, UInt_t idx);'.format(name = self.name))

        header.writeline('~{name}();'.format(name = self.name)) # destructor
        header.writeline('{name}& operator=({name} const&);'.format(name = self.name)) # assignment operator

        header.newline()

        header.writeline('static char const* typeName() {{ return "{name}"; }}'.format(name = self.name))

        header.newline()

        header.writeline('void print(std::ostream& = std::cout, UInt_t level = 1) const override;')
        header.writeline('void dump(std::ostream& = std::cout) const override;')

        header.newline()

        if len(self.functions) != 0:
            for function in self.functions:
                function.write_decl(header, context = 'class')
            header.newline()

        if self.is_singlet():
            context = 'Singlet'
        else:
            context = 'Element'

        has_public = False
        for ancestor in inheritance:
            if len(ancestor.branches) == 0:
                continue

            inherits_members = False

            for branch in ancestor.branches:
                if not hasattr(branch, 'refname') and branch.name.endswith('_'):
                    continue

                if not has_public:
                    has_public = True

                if not inherits_members:
                    if ancestor != self:
                        header.writeline('/* {name}'.format(name = ancestor.name))

                    inherits_members = True

                branch.write_decl(header, context = context)

            if inherits_members and ancestor != self:
                header.writeline('*/')

        if has_public:
            header.newline()

        has_protected = False
        for ancestor in inheritance:
            if len(ancestor.branches) == 0:
                continue

            inherits_members = False

            for branch in ancestor.branches:
                if hasattr(branch, 'refname') or not branch.name.endswith('_'):
                    continue

                if not has_protected:
                    header.indent -= 1
                    header.writeline('protected:')
                    header.indent += 1
                    has_protected = True

                if not inherits_members:
                    if ancestor != self:
                        header.writeline('/* {name}'.format(name = ancestor.name))

                    inherits_members = True

                branch.write_decl(header, context = context)

            if inherits_members and ancestor != self:
                header.writeline('*/')

        if has_protected:
            header.newline()
    
            header.indent -= 1
            header.writeline('public:')
            header.indent += 1

        header.write_custom_block('{name}.h.classdef'.format(name = self.name))

        header.newline()
        header.writeline('static utils::BranchList getListOfBranches();')

        if self.is_singlet():
            header.newline()
            header.indent -= 1
            header.writeline('protected:')
            header.indent += 1
            header.writeline('void doSetStatus_(TTree&, utils::BranchList const&) override;')
            header.writeline('utils::BranchList doGetStatus_(TTree&) const override;')
            header.writeline('utils::BranchList doGetBranchNames_(Bool_t) const override;')
            header.writeline('void doSetAddress_(TTree&, utils::BranchList const& = {"*"}, Bool_t setStatus = kTRUE) override;')
            header.writeline('void doBook_(TTree&, utils::BranchList const& = {"*"}) override;')
            header.writeline('void doInit_() override;')
            header.indent -= 1
        else:
            header.newline()
            header.writeline('void destructor(Bool_t recursive = kFALSE);')
            header.newline()
            header.indent -= 1
            header.writeline('protected:')
            header.indent += 1
            header.writeline('{name}(ArrayBase*);'.format(name = self.name))
            header.newline()
            header.writeline('void doBook_(TTree&, TString const&, utils::BranchList const& = {"*"}) override;')
            header.writeline('void doInit_() override;')
            header.indent -= 1

        header.writeline('};')

        header.newline()

        if not self.is_singlet():
            header.writeline('typedef Array<{name}> {name}Array;'.format(name = self.name))
            header.writeline('typedef Collection<{name}> {name}Collection;'.format(name = self.name))
            header.writeline('typedef Ref<{name}> {name}Ref;'.format(name = self.name))
            header.writeline('typedef RefVector<{name}> {name}RefVector;'.format(name = self.name))
            header.newline()

        header.write_custom_block('{name}.h.global'.format(name = self.name))
        header.newline()

        header.indent -= 1
        header.writeline('}')
        header.newline()

        header.writeline('#endif')
        header.close()

    def _write_method(self, out, context, methodspec, nestedcls = '', custom_block = False):
        """
        Util function to write class methods with a common pattern.
        """

        fname, rettype, arguments, generator, retexpr = methodspec[:5]
        try:
            pre_lines = methodspec[5]
        except:
            pre_lines = []
        try:
            parent_retval = methodspec[6]
        except:
            parent_retval = ''

        out.writeline(rettype)

        if fname.endswith(' const'):
            fname = fname[:-6]
            cqual = ' const'
        else:
            cqual = ''

        signature = []
        for arg in arguments:
            s = '{0} {1}'.format(*arg)
            if len(arg) == 3: # has default
                s += '/* = {0}*/'.format(arg[2])
                
            signature.append(s)

        args = ', '.join(arg[1] for arg in arguments)

        if nestedcls:
            name = self.name + '::' + nestedcls
            parent = self.parent + '::' + nestedcls
        else:
            name = self.name
            parent = self.parent

        out.writeline('{NAMESPACE}::{name}::{fname}({signature}){cqual}'.format(NAMESPACE = NAMESPACE, name = name, fname = fname, signature = ', '.join(signature), cqual = cqual))
        out.writeline('{')
        out.indent += 1
        if parent not in ['Singlet', 'Element', 'TreeEntry']:
            if parent_retval:
                out.writeline('{retval}({parent}::{fname}({args}));'.format(retval = parent_retval, parent = parent, fname = fname, args = args))
            else:
                out.writeline('{parent}::{fname}({args});'.format(parent = parent, fname = fname, args = args))
            out.newline()

        elif parent_retval:
            out.writeline('{retval};'.format(retval = parent_retval))
            out.newline()

        if len(pre_lines) != 0:
            for line in pre_lines:
                out.writeline(line)
            out.newline()

        if len(self.branches) != 0:
            for branch in self.branches:
                getattr(branch, generator)(out, context = context)

        if custom_block:
            out.newline()
            out.write_custom_block('{name}.cc.{method}'.format(name = name, method = fname))

        if retexpr:
            out.newline()
            out.writeline('return {expr};'.format(expr = retexpr))

        out.indent -= 1
        out.writeline('}')

    def generate_source(self):
        """
        Write the .cc file.
        """

        subst = {'NAMESPACE': NAMESPACE, 'name': self.name, 'parent': self.parent}

        src = FileOutput('{PACKDIR}/Objects/src/{name}.cc'.format(PACKDIR = PACKDIR, **subst))
        src.writeline('#include "../interface/{name}.h"'.format(**subst))

        if len(self.enums) != 0:
            src.newline()
            for enum in self.enums:
                enum.write_def(src, cls = '{NAMESPACE}::{name}'.format(**subst))

        if len(self.constants) != 0:
            src.newline()
            for constant in self.constants:
                constant.write_def(src, cls = '{NAMESPACE}::{name}'.format(**subst))

        src.newline()
        src.writeline('/*static*/')
        src.writeline('panda::utils::BranchList')
        src.writeline('{NAMESPACE}::{name}::getListOfBranches()'.format(**subst))
        src.writeline('{')
        src.indent += 1
        src.writeline('utils::BranchList blist;')
        if self.parent not in ['Singlet', 'Element']:
            src.writeline('blist += {parent}::getListOfBranches();'.format(**subst))
        if len(self.branches) != 0:
            src.writeline('blist += {{{bnames}}};'.format(bnames = ', '.join('"{name}"'.format(name = branch.name) for branch in self.branches if '!' not in branch.modifier)))
        src.writeline('return blist;')
        src.indent -= 1
        src.writeline('}')

        if self.is_singlet():
            src.newline()

            if self.instantiable:
                src.writeline('{NAMESPACE}::{name}::{name}(char const* _name/* = ""*/) :'.format(**subst))
                src.indent += 1
                initializers = ['{parent}(_name)'.format(**subst)]
                for branch in self.branches:
                    branch.init_default(initializers, context = 'Singlet')
                src.writelines(initializers, ',')
                src.indent -= 1
                src.writeline('{')
                src.indent += 1
                for branch in self.branches:
                    branch.write_default_ctor(src, context = 'Singlet')
                src.indent -= 1
                src.writeline('}')
                src.newline()
    
                src.writeline('{NAMESPACE}::{name}::{name}({name} const& _src) :'.format(**subst))
                src.indent += 1
                initializers = ['{parent}(_src.name_)'.format(**subst)]
                for branch in self.branches:
                    branch.init_copy(initializers, context = 'Singlet')
                src.writelines(initializers, ',')
                src.indent -= 1
                src.writeline('{')
                src.indent += 1
                for branch in self.branches:
                    branch.write_copy_ctor(src, context = 'Singlet')
                src.indent -= 1
                src.writeline('}')
                src.newline()

            src.writeline('{NAMESPACE}::{name}::~{name}()'.format(**subst))
            src.writeline('{')
            src.writeline('}')

            methods = [
                ('operator=', '{NAMESPACE}::{name}&'.format(**subst), [('{name} const&'.format(**subst), '_src')], 'write_assign', '*this'),
                ('doSetStatus_', 'void', [('TTree&', '_tree'), ('utils::BranchList const&', '_branches')], 'write_set_status', None),
                ('doGetStatus_ const', 'panda::utils::BranchList', [('TTree&', '_tree')], 'write_get_status', 'blist', [], 'utils::BranchList blist'),
                ('doSetAddress_', 'void', [('TTree&', '_tree'), ('utils::BranchList const&', '_branches', '{"*"}'), ('Bool_t', '_setStatus', 'kTRUE')], 'write_set_address', None),
                ('doBook_', 'void', [('TTree&', '_tree'), ('utils::BranchList const&', '_branches', '{"*"}')], 'write_book', None),
                ('doInit_', 'void', [], 'write_init', None)
            ]

            for method in methods:
                src.newline()
                self._write_method(src, 'Singlet', method, custom_block = (method[0] in ['operator=', 'doInit_']))

            src.newline()
            src.writeline('panda::utils::BranchList')
            src.writeline('{NAMESPACE}::{name}::doGetBranchNames_(Bool_t _fullName) const'.format(**subst))
            src.writeline('{')
            src.indent += 1
            src.writeline('if (_fullName)')
            src.writeline('  return getListOfBranches().fullNames(name_);')
            src.writeline('else')
            src.writeline('  return getListOfBranches().fullNames();')
            src.indent -= 1
            src.writeline('}')

        #if self.is_singlet():
        else:
            size_lines = ['TString size(_dynamic ? "[" + _name + ".size]" : TString::Format("[%d]", nmax_));']
            methods = [
                ('allocate', 'void', [('UInt_t', '_nmax')], 'write_allocate', None),
                ('deallocate', 'void', [], 'write_deallocate', None),
                ('setStatus', 'void', [('TTree&', '_tree'), ('TString const&', '_name'), ('utils::BranchList const&', '_branches')], 'write_set_status', None),
                ('getStatus const', 'panda::utils::BranchList', [('TTree&', '_tree'), ('TString const&', '_name')], 'write_get_status', 'blist', [], 'utils::BranchList blist'),
                ('setAddress', 'void', [('TTree&', '_tree'), ('TString const&', '_name'), ('utils::BranchList const&', '_branches', '{"*"}'), ('Bool_t', '_setStatus', 'kTRUE')], 'write_set_address', None),
                ('book', 'void', [('TTree&', '_tree'), ('TString const&', '_name'), ('utils::BranchList const&', '_branches', '{"*"}'), ('Bool_t', '_dynamic', 'kTRUE')], 'write_book', None, size_lines),
                ('releaseTree', 'void', [('TTree&', '_tree'), ('TString const&', '_name')], 'write_release_tree', None),
                ('resizeVectors_', 'void', [('UInt_t', '_size')], 'write_resize_vectors', None)
            ]

            for method in methods:
                src.newline()
                self._write_method(src, 'datastore', method, nestedcls = 'datastore')

            src.newline()
            src.newline()
            src.writeline('panda::utils::BranchList')
            src.writeline('{NAMESPACE}::{name}::datastore::getBranchNames(TString const& _name/* = ""*/) const'.format(**subst))
            src.writeline('{')
            src.indent += 1
            src.writeline('return {name}::getListOfBranches().fullNames(_name);'.format(**subst))
            src.indent -= 1
            src.writeline('}')

            if self.instantiable:
                src.newline()
                src.writeline('{NAMESPACE}::{name}::{name}(char const* _name/* = ""*/) :'.format(**subst))
                src.indent += 1
                initializers = ['{parent}(new {name}Array(1, _name))'.format(**subst)]
                for branch in self.branches:
                    branch.init_default(initializers, context = 'Element')
                src.writelines(initializers, ',')
                src.indent -= 1
                src.writeline('{')
                src.indent += 1
                for branch in self.branches:
                    branch.write_default_ctor(src, context = 'Element')
                src.indent -= 1
                src.writeline('}')

                src.newline()
                src.writeline('{NAMESPACE}::{name}::{name}({name} const& _src) :'.format(**subst))
                src.indent += 1
                initializers = ['{parent}(new {name}Array(1, _src.getName()))'.format(**subst)]
                for branch in self.branches:
                    branch.init_copy(initializers, context = 'Element')
                src.writelines(initializers, ',')
                src.indent -= 1
                src.writeline('{')
                src.indent += 1
                src.writeline('{parent}::operator=(_src);'.format(**subst))
                if len(self.branches) != 0:
                    src.newline()
                    for branch in self.branches:
                        branch.write_copy_ctor(src, context = 'Element')
                src.indent -= 1
                src.writeline('}')

            src.newline()
            src.writeline('{NAMESPACE}::{name}::{name}(datastore& _data, UInt_t _idx) :'.format(**subst))
            src.indent += 1
            initializers = ['{parent}(_data, _idx)'.format(**subst)]
            for branch in self.branches:
                branch.init_standard(initializers, context = 'Element')
            src.writelines(initializers, ',')
            src.indent -= 1
            src.writeline('{')
            src.indent += 1
            for branch in self.branches:
                branch.write_standard_ctor(src, context = 'Element')
            src.indent -= 1
            src.writeline('}')

            src.newline()
            src.writeline('{NAMESPACE}::{name}::{name}(ArrayBase* _array) :'.format(**subst))
            src.indent += 1
            initializers = ['{parent}(_array)'.format(**subst)]
            for branch in self.branches:
                branch.init_default(initializers, context = 'Element')
            src.writelines(initializers, ',')
            src.indent -= 1
            src.writeline('{')
            src.indent += 1
            for branch in self.branches:
                branch.write_default_ctor(src, context = 'Element')
            src.indent -= 1
            src.writeline('}')

            src.newline()
            src.writeline('{NAMESPACE}::{name}::~{name}()'.format(**subst))
            src.writeline('{')
            src.indent += 1
            src.writeline('destructor();')
            src.indent -= 1
            src.writeline('}')

            src.newline()
            src.writeline('void')
            src.writeline('{NAMESPACE}::{name}::destructor(Bool_t _recursive/* = kFALSE*/)'.format(**subst))
            src.writeline('{')
            src.indent += 1
            src.write_custom_block('{name}.cc.destructor'.format(**subst))
            if self.parent != 'Element':
                src.newline()
                src.writeline('if (_recursive)')
                src.indent += 1
                src.writeline('{parent}::destructor(kTRUE);'.format(**subst))
                src.indent -= 1
            src.indent -= 1
            src.writeline('}')

            methods = [
                ('operator=', '{NAMESPACE}::{name}&'.format(**subst), [('{name} const&'.format(**subst), '_src')], 'write_assign', '*this'),
                ('doBook_', 'void', [('TTree&', '_tree'), ('TString const&', '_name'), ('utils::BranchList const&', '_branches', '{"*"}')], 'write_book', None),
                ('doInit_', 'void', [], 'write_init', None)
            ]

            for method in methods:
                src.newline()
                self._write_method(src, 'Element', method, custom_block = (method[0] in ['operator=', 'doInit_']))

        src.newline()
        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::print(std::ostream& _out/* = std::cout*/, UInt_t _level/* = 1*/) const'.format(**subst))
        src.writeline('{')
        src.indent += 1
        src.write_custom_block('{name}.cc.print'.format(**subst), default = 'dump(_out);')
        src.indent -= 1
        src.writeline('}')

        src.newline()
        src.writeline('void')
        src.writeline('{NAMESPACE}::{name}::dump(std::ostream& _out/* = std::cout*/) const'.format(**subst))
        src.writeline('{')
        src.indent += 1

        if self.parent not in ['Singlet', 'Element']:
            src.writeline('{parent}::dump(_out);'.format(parent = self.parent))
            src.newline()

        if len(self.branches) != 0:
            for branch in self.branches:
                branch.write_dump(src)

        src.indent -= 1
        src.writeline('}')

        if len(self.functions):
            src.newline()
            for function in self.functions:
                function.write_def(src, context = self.name)

        src.newline()
        src.write_custom_block('{name}.cc.global'.format(**subst))

        src.close()
