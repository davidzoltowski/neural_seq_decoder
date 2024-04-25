
# modelName = 'Mamba_Run2_speckled_masking_standardized_v2'
# modelName = 'Mamba_Run2_speckled_masking_standardized_v3'
# modelName = 'Mamba_Run2_speckled_masking_standardized_v4'
# modelName = 'Mamba_Run2_speckled_masking_layerNorm_standardized_v5'
# modelName = 'Mamba_Run2_speckled_masking_layerNorm_standardized_bidirectional'
modelName = 'Mamba_Run2_speckled_masking_layerNorm_standardized_bidirectional_6layers'
# modelName = 'Mamba_Run2_speckled_masking_normalized'
# modelName = 'Mamba_Run2_normalized_without_speckled_masking'
# modelName = 'Mamba_Run2_normalized_without_speckled_masking_addCERLoss'

args = {}

args['outputDir'] = '/scratch/users/hdlee/speech_bci/logs/' + modelName
args['datasetPath'] = '/scratch/users/hdlee/speech_bci/competitionData/ptDecoder_ctc'
# args['datasetPath'] = '/scratch/users/hdlee/speech_bci/competitionData/ptDecoder_ctc_normalized'

# args['seqLen'] = 150
# args['maxTimeSeriesLen'] = 1200
# args['batchSize'] = 64
# args['lrStart'] = 1e-2 #0.02
# args['lrEnd'] = 1e-2 #0.02
# args['nUnits'] = 1024
# args['nBatch'] = 20000 #3000
# args['nLayers'] = 2
# args['seed'] = 15 # 0 
# args['nClasses'] = 40
# args['nInputFeatures'] = 256
# args['dropout'] = 0.4
# args['whiteNoiseSD'] = 0.8
# args['constantOffsetSD'] = 0.2
# args['gaussianSmoothWidth'] = 0.0
# args['strideLen'] = 1 # 4
# args['kernelLen'] = 1 # 32
# args['bidirectional'] = True
# args['l2_decay'] = 1e-5
# args["d_model"] = 1024 # 256
# args["d_state"] = 16
# args["d_conv"] = 4
# args["expand_factor"] = 1 # 4
# args['adamBeta2'] = 0.999 # could try 0.95
# args['nWarmup'] = 1
# args['cosine_anneal'] = True
# args['lrMin'] = 1e-6 # min for cosine annealing

# from neural_decoder.neural_decoder_trainer_mamba import trainModel

# trainModel(args)

args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 1e-2 #0.02
args['lrEnd'] = 1e-2 #1e-2 #0.02
args['nUnits'] = 1024
args['nBatch'] = 50000 # 20000
args['nLayers'] = 6 #1
args['seed'] = 15 # 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.0 #0.4

args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
# args['whiteNoiseSD'] = 0.0 
# args['constantOffsetSD'] = 0.0 

args['gaussianSmoothWidth'] = 0.0
args['strideLen'] = 1 # 4
args['kernelLen'] = 1 # 32
args['bidirectional_input'] = False
args['bidirectional'] = True
args['l2_decay'] = 1e-5 # orig = 1e-5
args["d_model"] = 1024
args["d_state"] = 16
args["d_conv"] = 4
args["expand_factor"] = 1 # 4
args['adamBeta2'] = 0.99
args['adamEPS'] = 1e-1
args['nWarmup'] = 1 # no warmup
args['cosine_anneal'] = False # constant lr
args['lrMin'] = 1e-6 # min for cosine annealing
args['clipGrad'] = 5.0 #1e2 # gradient clipping
args["speckled_masking_value"] = 0.0
args['speckled_mask_p'] = 0.3 #0.1 #0.05 #0.15 #0.35

from neural_decoder.neural_decoder_trainer_mamba import trainModel

trainModel(args)
