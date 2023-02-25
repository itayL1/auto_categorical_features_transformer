from enum import Enum


class CategoricalTransformationMethods(Enum):
    OneHot = 'one_hot'
    WeightOfEvidence = 'weight_of_evidence'
    LeaveOneOut = 'leave_one_out'
