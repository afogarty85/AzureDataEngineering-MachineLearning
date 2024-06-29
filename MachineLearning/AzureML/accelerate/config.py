

def calculate_embedding_sizes(cardinalities, min_size=10, max_size=50):
    """  
    Calculate embedding sizes based on the cardinalities of the categorical features.  

    Returns:  
    - A dictionary where the keys are feature names and the values are tuples (cardinality, embedding size).  
    """
    embedding_sizes = {}
    for feature, (cardinality, _) in cardinalities.items():
        # Apply the heuristic to the cardinality value
        emb_size = int(min(max_size, max(min_size, (cardinality // 2 + 1) ** 0.56)))
        embedding_sizes[feature] = (cardinality, emb_size)
    return embedding_sizes

# Determine categorical and numerical
numerical_cols = [
    'FromStateDurationMinutes', 'Wmi_CS_NumberOfLogicalProcessors', 'Wmi_CS_NumberOfProcessors', 'Wmi_CS_TotalPhysicalMemory',
    'HyperVCore_MinRoot', 'HyperVCore_PhysicalCoreCount',
    'HyperVCore_ActiveCoreCount', 'HyperVCore_SchedulerType', 'MemoryPartition_NumberofNumaNodes',
]
numerical_cols = [c + '_scaled' for c in numerical_cols]


binary_cols = ['IsLive', 'OOSDuringSpan']
housekeeping_cols = ['NodeId', 'Type', 'OOS_Span', 'ConcatSELText', 'LatestRecord']

label_col = ['IsRepeatOffender_SI']
group_keys = ['NodeId', 'OOS_Span']  
  
cat_cols = [
    'Tenant',
    'FromState',
    'ToState',
    'FaultCode',
    'RepeatedFaultCodes',
    'ProductName',
    'BaseboardVersion',
    'EnclosureVersion',
    'BIOSRelDate',
    'BIOSVersion',
    'SystemBIOSMajorVersion',
    'SystemBIOSMinorVersion',
    'BmcVersion',
    'EmbeddedControllerFirmwareMajorVersion',
    'EmbeddedControllerFirmwareMinorVersion',
    'CpldVersion',
    'OsImageName',
    'BladeFxVersion',
    'Wmi_CS_MOdel',
    'DatacenterId',
    'ClusterType',
    'Businessgroup',
    'RackSkuId',
    'Oem',
    'Generation',
    'BladeBiosSkuId',
    'MachinePoolType',
    'Capabilities',
]
cat_cols = [c + '_SI' for c in cat_cols]

text_col = 'ConcatSELText'

embedding_table_shapes = {'Tenant_SI': (11124, 25),
 'FromState_SI': (19, 25),
 'ToState_SI': (19, 25),
 'FaultCode_SI': (1075, 25),
 'RepeatedFaultCodes_SI': (287, 25),
 'ProductName_SI': (253, 25),
 'BaseboardVersion_SI': (212, 25),
 'EnclosureVersion_SI': (172, 25),
 'BIOSRelDate_SI': (416, 25),
 'BIOSVersion_SI': (971, 25),
 'SystemBIOSMajorVersion_SI': (5, 25),
 'SystemBIOSMinorVersion_SI': (12, 25),
 'BmcVersion_SI': (350, 25),
 'EmbeddedControllerFirmwareMajorVersion_SI': (15, 25),
 'EmbeddedControllerFirmwareMinorVersion_SI': (73, 25),
 'CpldVersion_SI': (154, 25),
 'OsImageName_SI': (1515, 25),
 'BladeFxVersion_SI': (548, 25),
 'Wmi_CS_MOdel_SI': (250, 25),
 'DatacenterId_SI': (405, 25),
 'ClusterType_SI': (14, 25),
 'Businessgroup_SI': (5, 25),
 'RackSkuId_SI': (1489, 25),
 'Oem_SI': (7, 25),
 'Generation_SI': (58, 25),
 'BladeBiosSkuId_SI': (576, 25),
 'MachinePoolType_SI': (9, 25),
 'Capabilities_SI': (10, 25)}

# calculate new feature size
embedding_table_shapes = calculate_embedding_sizes(embedding_table_shapes)

