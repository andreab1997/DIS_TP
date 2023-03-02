#! /bin/bash

MEM='250gb'
PYTHON='/project/theorie/gmagni/miniconda3/envs/n3lo_dis/bin/dis_tp'

function submit_job() {

    # run setup
    H_ID=$1
    ENTRY=$2
    NCORES=$3
    TYPE=$4
    PTO=$5
   
    COMMAND=$PWD'/launch_'$H_ID'-'$ENTRY'-'$PTO'-.sh'
    GRID_PATH=$PWD
    LAUNCH=$PYTHON' grids '$TYPE' '$H_ID' '$ENTRY' '$PTO' -n '$NCORES' -d '$GRID_PATH

    [ -e $COMMAND ] && rm $COMMAND
    echo $LAUNCH >> $COMMAND
    chmod +x $COMMAND

    # submission
    qsub -q smefit -W group_list=smefit -l nodes=1:ppn=$NCORES -l vmem=$MEM -l walltime=$WALLTIME $COMMAND
    # cleaning
    rm $COMMAND
}

# ########## MATCHING ##########
# charm
# submit_job '4' 'bq' '32' 'matching' '3'
# submit_job '4' 'bg' '32' 'matching' '3'

# # bottom
# submit_job '5' 'bq' '32' 'matching' '3'
# submit_job '5' 'bg' '32' 'matching' '3'

# # charm
# submit_job '4' 'bq' '32' 'matching' '2'
# submit_job '4' 'bg' '32' 'matching' '2'

# # bottom
# submit_job '5' 'bq' '32' 'matching' '2'
# submit_job '5' 'bg' '32' 'matching' '2'

########## TILDE ##########
# bottom
submit_job '5' '2q' '32' 'tilde' '2' 
submit_job '5' '2g' '32' 'tilde' '2'
submit_job '5' 'Lq' '32' 'tilde' '2'
submit_job '5' 'Lg' '32' 'tilde' '2'

# charm
submit_job '4' '2q' '32' 'tilde' '2'
submit_job '4' '2g' '32' 'tilde' '2'
submit_job '4' 'Lq' '32' 'tilde' '2'
submit_job '4' 'Lg' '32' 'tilde' '2'


# bottom
# submit_job '5' '2q' '32' 'tilde' '3'
# submit_job '5' '2g' '32' 'tilde' '3'
# submit_job '5' 'Lq' '32' 'tilde' '3'
# submit_job '5' 'Lg' '32' 'tilde' '3'

# # # charm
# submit_job '4' '2q' '32' 'tilde' '3'
# submit_job '4' '2g' '32' 'tilde' '3'
# submit_job '4' 'Lq' '32' 'tilde' '3'
# submit_job '4' 'Lg' '32' 'tilde' '3'