
# ==============================================================================
# Header
# ==============================================================================
#  Author: Matthewsmith12@cmail.carleton.ca, derived from stmenary@cern.ch and tutorial code by Hannes Mildner at https://gitlab.cern.ch/eft-tools/smeft-jos/blob/master/EFTMCGeneration.md

# to submit a job, type:
# 'condor_submit submit_file.condor'

# check on job status with:
# condor_q  or condor_q -better-analyze
# condor_q -dag
# delete a job: condor_rm job_id

# ==============================================================================
# Imports
# ==============================================================================
import sys, os, shutil
import subprocess
import argparse 


# ==============================================================================
# Helper Functions
# ==============================================================================

#  ---------------------------print_message-------------------------------------
#  Print a message to the screen
def print_message (cat, message) :
    print("{}    Matt_Condor_Script.py    {}".format(cat, message))

 
#  ------------------------------parse_args-------------------------------------
#  Parse the command-line arguments using argparse module
def parse_args () :
    parser = argparse.ArgumentParser(description="Run a MG5 job for a given SMEFT coefficient")

    parser.add_argument("--identifier"        , type=str           , help="Identifier used to distinguish separate runs"         , default=""    )
    parser.add_argument("--verbose"           , action="store_true", help="Be verbose"                                           , default=False )

    parser.add_argument("--file"              , type=str           , help="Comma-separated list of input ROOT files"             , default=""    )
    parser.add_argument("--weight_file"       , type=str           , help="Weight file (or 'None')"                              , default="None")
    parser.add_argument("--weight_names"      , type=str           , help="Comma-separated list of weight names"                 , default=""    )
    parser.add_argument("--nEns"              , type=int           , help="Number of ensembles"                                  , default=0     )
    parser.add_argument("--outFile"           , type=str           , help="Output ROOT file"                                     , default=""    )
    parser.add_argument("--truth"             , action="store_true", help="Enable truth mode"                                    , default=False )
    parser.add_argument("--do_IBU"            , action="store_true", help="Enable IBU branch creation"                           , default=False )
    parser.add_argument("--is_data"           , action="store_true", help="Dissable truth input reading"                         , default=False )
    parser.add_argument("--maxEvents"         , type=int           , help="Max number of events to process"                      , default=5000000 )
    parser.add_argument("--kinematic_region"  , type=int           , help="Kinematic region index"                               , default=0     )
    parser.add_argument("--track_variations"  , type=str           , help="Comma-separated list of track systematic variations"  , default=""    )

    return parser.parse_args()


#  -------------------------------configure-------------------------------------
#  Convert the argparse arguments into an internal settings dictionary
def configure (args=None) :
    if type(args) == type(None) :
        args = parse_args()
    settings = {}
    settings["identifier"        ] = str(args.identifier)
    settings["verbose"           ] = bool(args.verbose)

    if args.file :
        settings["fileNames"     ] = [ f.strip() for f in args.file.split(",") if f.strip() ]
    else :
        settings["fileNames"     ] = []

    # Weight file
    settings["weight_file"       ] = str(args.weight_file)

    # Weight names (comma-separated)
    if args.weight_names :
        settings["weight_names"  ] = [ w.strip() for w in args.weight_names.split(",") if w.strip() ]
    else :
        settings["weight_names"  ] = []

    # Track systematic variations
    if args.track_variations :
        variations = [""]
        for t in args.track_variations.split(",") :
            variations.append( t.strip() + "_" )
        settings["trackVariations"] = variations
    else :
        settings["trackVariations"] = [ "", "syst_pTScale", "syst_Fake", "syst_TrackFilter", "syst_JetTrackFilter"]

    # Ensemble logic
    if args.weight_file == "None" :
        settings["nEns"          ] = 0
    else :
        settings["nEns"          ] = int(args.nEns)

    # Other simple fields
    settings["outFile"           ] = str(args.outFile)
    settings["isTruth"           ] = bool(args.truth)
    settings["do_IBU"            ] = bool(args.do_IBU)
    settings["maxEvents"         ] = int(args.maxEvents)
    settings["kinematic_region"  ] = int(args.kinematic_region)

    return settings



#  -----------------------------Make_Folder-------------------------------------
# smal helper fn to cleanup code.  checks whether folder/directory exists
def Make_Folder(folder_name, kill_code):
    try:
        os.mkdir(folder_name)
        print_message ("INFO", "folder '{}' created ".format(folder_name))

    except:
        print_message ("INFO", "folder '{}' already exists".format(folder_name))
        if kill_code:
            exit(2)
    return

#  -----------------------------build_doHisto_command-------------------------------------
# converts the inputs, which are in the settings dictionary, into a ./doHisto.out command
def build_doHisto_command(settings) :

    cmd = "./doHisto.out"

    # --file (comma-joined list)
    if settings["fileNames"] :
        cmd += " --file " + ",".join(settings["fileNames"])

    # --weight_file
    if settings["weight_file"] != "None" :
        cmd += " --weight_file " + settings["weight_file"]

    # --weight_names
    if len(settings["weight_names"]) > 0 :
        cmd += " --weight_names " + ",".join(settings["weight_names"])

    # --nEns
    if settings["nEns"] > 0 :
        cmd += f" --nEns {settings['nEns']}"

    # --truth
    if settings["isTruth"] :
        cmd += " --truth"

    # --do_IBU
    if settings["do_IBU"] :
        cmd += " --do_IBU"
        
    # --is_data
    if settings["is_data"] :
        cmd += " --is_data"

    # --track_variations
    if len(settings["trackVariations"]) > 1 :
        # skip the first "" nominal
        systs = [v[:-1] for v in settings["trackVariations"][1:]]
        cmd += " --track_variations " + ",".join(systs)

    # --maxEvents
    if settings["maxEvents"] != 5000000 :
        cmd += f" --maxEvents {settings['maxEvents']}"

    # --kinematic_region
    cmd += f" --kinematic_region {settings['kinematic_region']}"

    # --outFile
    if settings["outFile"] :
        cmd += " --outFile " + settings["outFile"]

    return cmd


#  -----------------------------setup_job_area----------------------------------
# setup / Prep directory to recieve condor files
def setup_job_area (settings, verbose) :
    Make_Folder("logs", False)
    Make_Folder("logs/" + settings["identifier"], False)
    Make_Folder("logs/{}".format(settings["identifier"]) , False)
    Make_Folder("Condor/Job_Files/" + settings["identifier"], False)
    return
    #
    # try:
    #     shutil.copyfile("EW_Fit.cxx", working_dir+"/EW_Fit.cxx")
    # except:
    #     if verbose : print_message("INFO", "Not copied EW_Fit.cxx into working directory")


#  ---------------------------------runJob--------------------------------------
# makes executable and submit file for condor. Calls condor_submit.
def runJob ( settings, verbose) :
    cwd = os.getcwd()
    run_job_sh_fname = "Condor/Job_Files/{}/Condor_Exe_File.sh".format(settings["identifier"])
    run_job_sh = open(run_job_sh_fname, "w")
    run_job_sh.write("#!/bin/bash\n\n")
    run_job_sh.write("echo test 1.1\n")
    run_job_sh.write("export ATLAS_LOCAL_ROOT_BASE=\"/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase\"\n")
    run_job_sh.write("echo test 1.2\n")
    run_job_sh.write("setupATLAS\n")
    run_job_sh.write("lsetup \"views LCG_108 x86_64-el9-gcc15-opt\"\n")
    run_job_sh.write("echo test 1.3\n")
    run_job_sh.write("export OMP_NUM_THREADS=8\n")
    run_job_sh.write("echo \"Running with $OMP_NUM_THREADS threads\"\n")
    run_job_sh.write("cd {}\n".format(cwd))
    run_job_sh.write("make\n")
    cmd = build_doHisto_command(settings)
    run_job_sh.write(cmd + "\n")
    run_job_sh.write("echo test4\n")
    run_job_sh.close()

    if verbose :
        print_message("\nINFO", "The following is the condor executable file named {}: \n".format(run_job_sh_fname))
        subprocess.call(["cat", run_job_sh_fname])
        print_message("\nINFO", "Making {} executable.".format(run_job_sh_fname))
        subprocess.call(["chmod", "777", run_job_sh_fname])

    submit_job_sh_fname = "Condor/Job_Files/{}/Condor_Submit_File.condor".format(settings["identifier"])
    submit_job_sh = open(submit_job_sh_fname, "w")
    submit_job_sh.write("executable   = Condor/Job_Files/{}/Condor_Exe_File.sh \n".format(settings["identifier"]) )
    submit_job_sh.write("arguments    = $(Process)\n")
    submit_job_sh.write("universe     = vanilla\n")
    submit_job_sh.write("output       = logs/{}/out_file.out\n".format(settings["identifier"]) )
    submit_job_sh.write("error        = logs/{}/err_file.err\n".format(settings["identifier"]) )
    submit_job_sh.write("log          = logs/{}/log_file.err\n".format(settings["identifier"]) )
    submit_job_sh.write("getenv       = True\n")
    submit_job_sh.write("request_cpus = 8\n")
    submit_job_sh.write("request_memory = 32 GB\n")
    submit_job_sh.write("+JobFlavour  = \"nextweek\"\n")
    submit_job_sh.write("queue\n")
    submit_job_sh.close()

    if verbose :
        print_message("\nINFO", "The following is the condor submit file named {}:\n".format(submit_job_sh_fname))
        subprocess.call(["cat", submit_job_sh_fname])
        print_message("\nINFO", "Submitting job")

    subprocess.check_call(["condor_submit", submit_job_sh_fname], stdout=sys.stdout, stderr=sys.stdout)




# ==============================================================================
# Main
# ==============================================================================
#  Fallback - run as a script when this module is called as __main__
#
if __name__ == "__main__" :

    args = parse_args()

    settings = configure(args)

    if "identifier" not in settings:
        raise RuntimeError("No identifier provided in settings")

    cwd = os.getcwd()
    setup_job_area(settings, args.verbose)

    #os.chdir(settings["outfile_path"])
    runJob(settings, args.verbose)

    #os.chdir("..")
















    #
