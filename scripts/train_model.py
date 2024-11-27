
# modelName = 'speechBaseline_seedModel_24InputLayers_v2'
# modelName = 'speechBaseline_seedModel_24InputLayers_v3'
modelName = 'speechBaseline_allData_v3'
# modelName = 'speechBaseline_seedModel_24InputLayers_rotationAug'

args = {}
args['outputDir'] = '/scratch/users/hdlee/speech_bci/logs/' + modelName
# args['datasetPath'] = '/scratch/users/hdlee/speech_bci/competitionData/ptDecoder_ctc_seedData'
args['datasetPath'] = '/scratch/users/hdlee/speech_bci/competitionData/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.025 #0.02
args['lrEnd'] = 0.0005 #0.00 #0.02
args['nUnits'] = 1024
args['nBatch'] = 20000 #10000 #6500 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5
# args["speckled_masking_value"] = 0.0
args['speckled_mask_p'] = 0.3

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
