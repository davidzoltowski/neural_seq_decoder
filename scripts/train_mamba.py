# modelName = 'Mamba_Run5'
# modelName = 'Mamba_Run7'
modelName = 'Mamba_Run_leaderboard_060124'
# modelName = 'Complex_Mamba_Run_leaderboard_060124'
# modelName = 'Complex_Mamba_Run_leaderboard_070924'
# modelName = 'Complex_Mamba_Run_leaderboard_070924_v2'

args = {}

args['outputDir'] = '/scratch/users/hdlee/speech_bci/logs/' + modelName
args['datasetPath'] = '/scratch/users/hdlee/speech_bci/competitionData/ptDecoder_ctc'
# args['datasetPath'] = '/scratch/users/hdlee/speech_bci/competitionData/ptDecoder_ctc_normalized'

args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 1e-2 #1e-4
args['lrEnd'] = 1e-2 #1e-4
args['nUnits'] = 1024
args['nBatch'] = 8000 #50000 #12000 #8000 #20000
args['nLayers'] = 6
args['seed'] = 15
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.45

args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2

args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional_input'] = False
args['bidirectional'] = True
args['l2_decay'] = 1e-5 # orig = 1e-5
args["d_model"] = 1024 #2048 #1024
args["d_state"] = 16
args["d_conv"] = 4
args["expand_factor"] = 1
args['adamBeta2'] = 0.99
args['adamEPS'] = 1e-1
args['nWarmup'] = 1 # no warmup
args['cosine_anneal'] = False # constant lr
args['lrMin'] = 1e-6 # min for cosine annealing
args['clipGrad'] = 0.0 #1e2 # gradient clipping
args["renormalize_masking"] = False #True #False
args["masking_value"] = 0.0
# original
args["feature_mask_n"] = 1
args["feature_mask_max_len"] = 2
args["time_mask_n"] = 1
args["time_mask_max_len"] = 2
args['speckled_mask_p'] = 0.45

# step LR
args['step_batch'] = 6101
args['step_gamma'] = 0.01
args["step_feature_mask_n"] = 4
args["step_feature_mask_max_len"] = 4
args["step_time_mask_n"] = 4
args["step_time_mask_max_len"] = 4

args["time_warp_W"] = 0

args['modelWeightPath'] = '' # f'/scratch/users/hdlee/speech_bci/logs/Mamba_Run9' + "/modelWeights"

from neural_decoder.neural_decoder_trainer_mamba import trainModel

trainModel(args)
