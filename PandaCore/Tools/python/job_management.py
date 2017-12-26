#!/usr/bin/env python
'''@package docstring
'''
import time
from re import sub
from sys import exit 
import cPickle as pickle
from collections import namedtuple
from condor import classad,htcondor
from Misc import PInfo,PDebug,PWarning,PError
from os import getenv,getuid,system,path,environ

# module was partitioned to facilitate reading job configs
# on nodes that do not have htcondor bindings
from job_config import * 


#############################################################
# HTCondor interface for job submission and tracking
#############################################################

job_status = {
           1:'idle',
           2:'running',
           3:'removed',
           4:'completed',
           5:'held',
           6:'transferring output',
           7:'suspended',
        }

base_job_properties = None 
pool_server = None 
schedd_server = getenv('HOSTNAME')
should_spool = False
query_owner = getenv('USER')
try:
    urgent = int(getenv('SUBMIT_URGENT'))
    if urgent:
        acct_grp_t3 = 'group_t3mit.urgent'
    else:
        acct_grp_t3 = 'group_t3mit'
except:
    acct_grp_t3 = 'group_t3mit'

def setup_schedd(config='T3'):
    global pool_server, schedd_server, base_job_properties, should_spool
    if config=='T3':
        base_job_properties = {
   "Cmd" : "WORKDIR/exec.sh",
            "WhenToTransferOutput" : "ON_EXIT",
            "ShouldTransferFiles" : "YES",
            "X509UserProxy" : "/tmp/x509up_uUID",
            #"OnExitHold" : classad.ExprTree("( ExitBySignal == true ) || ( ExitCode != 0 )"),
            "RequestMemory" : 2100,
            "In" : "/dev/null",
            "TransferInput" : "WORKDIR/cmssw.tgz,WORKDIR/skim.py,WORKDIR/x509up",
        }

        pool_server = None
        schedd_server = getenv('HOSTNAME')
        should_spool = False
        query_owner = getenv('USER')
    elif config=='SubMIT':
        base_job_properties = {
            "Cmd" : "WORKDIR/exec.sh",
            "WhenToTransferOutput" : "ON_EXIT",
            "ShouldTransferFiles" : "YES",
            "Requirements" : 
                classad.ExprTree('( ( ( OSGVO_OS_STRING == "RHEL 6" && HAS_CVMFS_cms_cern_ch ) || GLIDEIN_REQUIRED_OS == "rhel6" || ( GLIDEIN_Site == "MIT_CampusFactory" && ( BOSCOGroup == "bosco_cms" ) && HAS_CVMFS_cms_cern_ch ) ) && ( isUndefined(GLIDEIN_Entry_Name) ||  !stringListMember(GLIDEIN_Entry_Name,"CMS_T2_US_Nebraska_Red_op,CMS_T2_US_Nebraska_Red_gw1_op,CMS_T2_US_Nebraska_Red_gw2_op,CMS_T3_MX_Cinvestav_proton_work,CMS_T3_US_Omaha_tusker,CMSHTPC_T3_US_Omaha_tusker,Glow_US_Syracuse_condor,Glow_US_Syracuse_condor-ce01,Gluex_US_NUMEP_grid1,HCC_US_BNL_gk01,HCC_US_BNL_gk02,HCC_US_BU_atlas-net2,OSG_US_FIU_HPCOSGCE,OSG_US_Hyak_osg,OSG_US_UConn_gluskap,OSG_US_SMU_mfosgce",",") ) && ( isUndefined(GLIDEIN_Site) ||  !stringListMember(GLIDEIN_Site,"SU-OG,HOSTED_BOSCO_CE",",") ) ) && ( ( Arch == "INTEL" || Arch == "X86_64" ) ) && ( TARGET.OpSys == "LINUX" ) && ( TARGET.Disk >= RequestDisk ) && ( TARGET.Memory >= RequestMemory ) && ( TARGET.HasFileTransfer )'),
            "AcctGroup" : "analysis",
            "AccountingGroup" : "analysis.USER",
            "X509UserProxy" : "/tmp/x509up_uUID",
            "OnExitHold" : classad.ExprTree("( ExitBySignal == true ) || ( ExitCode != 0 )"),
            "In" : "/dev/null",
            "TransferInput" : "WORKDIR/cmssw.tgz,WORKDIR/skim.py,WORKDIR/x509up",
            "ProjectName" : "CpDarkMatterSimulation",
            "Rank" : "Mips",
            'SubMITOwner' : 'USER',
        }

        pool_server = 'submit.mit.edu:9615'
        schedd_server ='submit.mit.edu'
        query_owner = 'anonymous'
        should_spool = True
    else:
        PError('job_management.setup_schedd','Unknown config %s'%config)
        raise ValueError

schedd_config = getenv('SUBMIT_SCHEDD')
schedd_config = 'T3' if not schedd_config else schedd_config
setup_schedd(schedd_config) 


def environ_to_condor():
    s = '' 
    for k,v in environ.iteritems():
        if any([x in k for x in ['PANDA','SUBMIT','USER']]):
            s += '%s=%s '%(k,v)
    return s 


class _BaseSubmission(object):
    def __init__(self, cache_filepath):
        self.cache_filepath = cache_filepath
        try: # figure out which (re-)submission attempt this is
            with open(cache_filepath,'rb') as fcache:
                self.sub_id = len(pickle.load(fcache))
        except:
            self.sub_id = 0
        self.submission_time = -1
        self.cluster_id = None # HTCondor ClusterID
        self.proc_ids = None # ProcID of each job
        if pool_server:
            self.coll = htcondor.Collector(pool_server)
        else:
            self.coll = htcondor.Collector()
        self.schedd = htcondor.Schedd(self.coll.locate(htcondor.DaemonTypes.Schedd,
                                                       schedd_server))
        self.custom_job_properties = {}


    def query_status(self):
        if not self.cluster_id:
            PError(self.__name__+".status",
                   "This submission has not been executed yet (ClusterId not set)")
            raise RuntimeError
        results = self.schedd.query(
            'Owner =?= "%s" && ClusterId =?= %i'%(query_owner,self.cluster_id))
        jobs = {x:[] for x in ['running','idle','held','other']}
        for job in results:
            proc_id = int(job['ProcId'])
            status = job['JobStatus']
            try:
                if type(self.arguments)==dict:
                    samples = [self.arguments[self.proc_ids[proc_id]]]
                else:
                    samples = self.proc_ids[proc_id].split()
            except KeyError:
                continue # sometimes one extra dummy job is created and not tracked, oh well
            if job_status[status] in jobs:
                jobs[job_status[status]] += samples
            else:
                jobs['other'] += samples
        return jobs


    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['coll']
        del odict['schedd']
        return odict


    def __setstate__(self,odict):
        self.__dict__.update(odict)
        self.coll = htcondor.Collector(pool_server)
        self.schedd = htcondor.Schedd(self.coll.locate(htcondor.DaemonTypes.Schedd,
                                                       schedd_server))


    def save(self):
        try: 
            with open(self.cache_filepath,'rb') as fcache:
                cache = pickle.load(fcache)
        except:
            cache = []
        cache.append(self)
        with open(self.cache_filepath,'wb') as fcache:
            pickle.dump(cache,fcache,2)


class SimpleSubmission(_BaseSubmission):
    def __init__(self,cache_dir,executable=None,arglist=None,arguments=None,nper=1):
        super(SimpleSubmission,self).__init__(cache_dir+'/submission.pkl')
        self.__name__='SimpleSubmission'
        self.cache_dir = cache_dir
        if executable!=None:
            self.executable = executable
            self.arguments = arguments
            self.arglist = arglist
            self.nper = nper
        else:
            try:
                pkl = pickle.load(open(self.cache_filepath))
                last_sub = pkl[-1]
                self.executable = last_sub.executable
                self.arguments = last_sub.arguments
                self.arglist = last_sub.arglist
                self.nper = last_sub.nper
            except:
                PError(self.__name__+'.__init__',
                       'Must provide a valid cache or arguments!')
                raise RuntimeError
        self.cmssw = getenv('CMSSW_BASE')
        self.workdir = cache_dir + '/workdir/'
        self.logdir = cache_dir + '/logdir/'
        for d in [self.workdir,self.logdir]:
            system('mkdir -p '+d)
        if type(self.arglist)==list:
            with open(cache_dir+'/workdir/args.list','w') as fargs:
                fargs.write('\n'.join(self.arglist))
            if not self.arguments:
                self.arguments = range(1,len(self.arglist)+1)
            self.arglist = cache_dir+'/workdir/args.list'
    def execute(self,njobs=None):
        self.submission_time = time.time()
        runner = '''
#!/bin/bash
env
cd {0} 
eval `/cvmfs/cms.cern.ch/common/scramv1 runtime -sh`
cd -
for i in $@; do
    arg=$(sed "${{i}}q;d" {3}) # get the ith line
    echo $arg
    {1} $arg && echo $i >> {2};
done'''.format(self.cmssw,self.executable,self.workdir+'/progress.log',self.arglist)
        with open(self.workdir+'exec.sh','w') as frunner:
            frunner.write(runner)
        repl = {'WORKDIR' : self.workdir,
                'LOGDIR' : self.logdir,
                'UID' : str(getuid()),
                'USER' : getenv('USER'),
                'SUBMITID' : str(self.sub_id)}
        cluster_ad = classad.ClassAd()

        job_properties = base_job_properties.copy()
        #for k in ['X509UserProxy','TransferInput']:
        for k in ['TransferInput','ShouldTransferFiles','WhenToTransferOutput']:
            del job_properties[k]
        # job_properties['Environment'] = 'A=B C=D'
        job_properties['Environment'] = environ_to_condor()
        for key,value in job_properties.iteritems():
            if type(value)==str and key!='Environment':
                for pattern,target in repl.iteritems():
                    value = value.replace(pattern,target)
            cluster_ad[key] = value

        proc_properties = {
            'UserLog' : 'LOGDIR/SUBMITID_PROCID.log',
            'Out' : 'LOGDIR/SUBMITID_PROCID.out',
            'Err' : 'LOGDIR/SUBMITID_PROCID.err',
        }
        proc_id=0
        procs = []
        self.arguments = sorted(self.arguments)
        n_to_run = len(self.arguments)/self.nper+1
        arg_mapping = {} # condor arg -> job args
        for idx in xrange(n_to_run):
            if njobs and proc_id>=njobs:
                break
            repl['PROCID'] = '%i'%idx
            proc_ad = classad.ClassAd()
            for key,value in proc_properties.iteritems():
                if type(value)==str:
                    for pattern,target in repl.iteritems():
                        value = value.replace(pattern,target)
                proc_ad[key] = value
            proc_ad['Arguments'] = ' '.join(
                    [str(x) for x in self.arguments[self.nper*idx:min(self.nper*(idx+1),len(self.arguments))]]
                    )
            arg_mapping[idx] = proc_ad['Arguments']
            procs.append((proc_ad,1))
            proc_id += 1

        PInfo(self.__name__+'.execute','Submitting %i jobs!'%(len(procs)))
        self.submission_time = time.time()
        results = []
        self.proc_ids = {}
        if len(procs):
            PInfo(self.__name__+'.execute','Cluster ClassAd:','')
            print cluster_ad 
            self.cluster_id = self.schedd.submitMany(cluster_ad, procs, spool=should_spool, ad_results=results)
            if should_spool:
                self.schedd.spool(results)
            for result,idx in zip(results,range(n_to_run)):
                self.proc_ids[int(result['ProcId'])] = arg_mapping[idx]
            PInfo(self.__name__+'.execute','Submitted to cluster %i'%(self.cluster_id))
        else:
            self.cluster_id = -1

    def check_missing(self, only_failed=True):
        try:
            finished = map(lambda x : int(x.strip()), [x for x in  open(self.workdir+'/progress.log').readlines() if len(x.strip())])
        except IOError:
            finished = []
        status = self.query_status()
        for k,v in status.iteritems():
            status[k] = [int(x) for x in v]
        missing = set([]); done = set([]); running = set([]); idle = set([])
        for idx in self.arguments:
            args = str(idx)
            if idx in finished:
                done.add(idx)
                continue 
            if only_failed and (idx in status['running']):
                running.add(idx)
                continue 
            if only_failed and (idx in status['idle']):
                idle.add(idx)
                continue 
            missing.add(idx)
        return missing, done, running, idle


class Submission(_BaseSubmission):
    '''Submission 
    
    This class is used specifically for heavy analysis-specific submission
    with robust re-packaging and re-submission of failures.
    
    Extends:
        _BaseSubmission
    '''
    def __init__(self,sample_configpath,cache_filepath):
        super(Submission,self).__init__(cache_filepath)
        self.__name__='Submission'
        self.arguments = read_sample_config(sample_configpath)
        self.configpath = sample_configpath
        

    def execute(self,njobs=None):
        logdir = getenv('SUBMIT_LOGDIR')
        workdir = getenv('SUBMIT_WORKDIR')
        repl = { 'LOGDIR':logdir,
                 'WORKDIR':workdir,
                 'UID':str(getuid()),
                 'USER':getenv('USER'),
                 'SUBMITID':str(self.sub_id), }
        cluster_ad = classad.ClassAd()
        job_properties = base_job_properties.copy()
        job_properties['TransferInput'] += ',%s'%(self.configpath)
        for key,value in job_properties.iteritems():
            if type(value)==str:
                for pattern,target in repl.iteritems():
                    value = value.replace(pattern,target)
            cluster_ad[key] = value
        for key,value in self.custom_job_properties.iteritems():
            if type(value)==str:
                for pattern,target in repl.iteritems():
                    value = value.replace(pattern,target)
            cluster_ad[key] = value
        PInfo(self.__name__+'.execute','Cluster ClassAd:','')
        print cluster_ad

        proc_properties = {
            "Arguments" : "PROCID SUBMITID",
            'UserLog' : 'LOGDIR/SUBMITID.log',
            'Out' : 'LOGDIR/SUBMITID_PROCID.out',
            'Err' : 'LOGDIR/SUBMITID_PROCID.err',
        }
        proc_id=0
        procs = []
        for name in sorted(self.arguments):
            sample = self.arguments[name]
            repl['PROCID'] = '%i'%sample.get_id() #name.split('_')[-1] 
            proc_ad = classad.ClassAd()
            for key,value in proc_properties.iteritems():
                if type(value)==str:
                    for pattern,target in repl.iteritems():
                        value = value.replace(pattern,target)
                proc_ad[key] = value
            procs.append((proc_ad,1))
            proc_id += 1
            if njobs and proc_id>=njobs:
                break

        PInfo('Submission.execute','Submitting %i jobs!'%(len(procs)))
        self.submission_time = time.time()
        results = []
        self.cluster_id = self.schedd.submitMany(cluster_ad, procs, spool=should_spool, ad_results=results)
        if should_spool:
            PInfo('Submission.execute','Spooling inputs...')
            self.schedd.spool(results)
        self.proc_ids = {}
        for result,name in zip(results,sorted(self.arguments)):
            self.proc_ids[int(result['ProcId'])] = name
#            print 'Mapping %i->%s'%(result['ProcId'],name)
        PInfo('Submission.execute','Submitted to cluster %i'%(self.cluster_id))

