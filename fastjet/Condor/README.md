All support files required to run this code through condor using the "Run_On_Condor.cpp" macro are conveniently held here. Currently it will only run the code "Generate_BDT.cpp"macro. The files include:

    1. Matt_Condor_Script.py
        - This will call the command 'condor_submit submit_file.condor' and create two associated files to be placed in this directory: Condor_Exe_File.sh and Condor_Submit_File.condor.

    2. The executable file Condor_Exe_File.sh.
        - This file downloads root and executs the code within the condor environment

    3. Condor_Submit_File.condor.
        - Specifies submit parameters to condor

    4. Dylan_Condor_Script.py
        - this is what my script is based off of.


For later:
Submit file to condor

 we will use the file Condor_Run_Job.sh to set up the Root environment in condor and to execute the run calls for our code

to submit a job, type: 'condor_submit submit_file.condor'

check on job status with:
    - condor_q  or condor_q -better-analyze
    - delete a job: condor_rm job_id
