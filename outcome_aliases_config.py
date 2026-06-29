"""
结局指标别名配置文件
用于Meta分析中的结局指标匹配
"""

# ==================== 认知功能相关指标 ====================
COGNITIVE_OUTCOMES = {
    # MMSE相关 - 添加更多变体
    "mmse": [
        "mini mental state examination",
        "mini-mental state examination", 
        "mini mental state exam",
        "mmse",
        "mmse score",
        "mmse total score",
        "folstein",
        "folstein mmse"
    ],
    
    # MoCA相关
    "moca": [
        "montreal cognitive assessment",
        "montreal cognitive",
        "moca",
        "moca score",
        "moca total score"
    ],
    
    # ADAS-Cog相关 - 添加更多变体
    "adas-cog": [
        "alzheimer disease assessment scale cognitive",
        "alzheimer disease assessment scale",
        "adas cognitive",
        "adas-cog",
        "adascog",
        # "adas-cog-11",
        # "adas-cog-13",
        "adas cog",
        "adas cognitive subscale"
    ],
    
    # CDR相关
    "cdr": [
        "clinical dementia rating",
        "cdr",
        "cdr-sb",
        "cdr sum of boxes"
    ]
}

# ==================== 抑郁相关指标 ====================
OCD_OUTCOMES = {
    "ybocs": [
        "yale-brown obsessive compulsive scale",
        "yale brown obsessive compulsive scale",
        "y-bocs",
        "ybocs",
        # "y-bocs-10",
        # "ybocs-10",
        # "y-bocs-18",
        # "ybocs-18",
        "yale-brown obsessive compulsive scale-second edition",
        # "y-bocs-ii",
        # "ybocs-ii",
        "yale-brown obsessive compulsive rating scale",
        "brown obsessive compulsive scale",
        "yale obsessive compulsive scale",
        "yocs"
    ],
    "oci-r": [
        "obsessive-compulsive inventory-revised",
        "obsessive compulsive inventory revised",
        "obsessive-compulsive inventory",
        "obsessive compulsive inventory",
        "oci-r",
        "ocir",
        "oci-r-18",
        "obsessive compulsive inventory-revised",
        "obsessive-compulsive inventory - reduced",
        "obsessive compulsive inventory - reduced",
        "oci-r"
    ],
    "docs": [
        "dimensional obsessive-compulsive scale",
        "dimensional obsessive compulsive scale",
        # "docs-20",
        # "docs-10",
        # "docs-5",
        "docs short form",
        "docs brief form",
        "docs"
    ],
    "bai": [
        "beck anxiety inventory",
        "beck anxiety inventory-ii",
        "beck anxiety scale",
        "bai-21",
        "bai-fs",
        "beck anxiety inventory - fast screen",
        "beck anxiety index",
        "bai"
    ]
}





# ==================== 抑郁相关指标 ====================
DEPRESSION_OUTCOMES = {
    # HAMD相关
    "hamd": [
        "hamilton depression rating scale",
        "hamilton depression scale",
        "hamilton depression score",
        "ham-d",
        "hamd",
        "hamd-17",
        "hamd-24",
        "hamilton rating scale for depression",
        "hdrs",
        "hdrs-17",
        "hdrs-24"
    ],
    
    # MADRS相关
    "madrs": [
        "montgomery asberg depression rating scale",
        "montgomery-asberg depression rating scale",
        "madrs"
    ],
    
    # BDI相关
    "bdi": [
        "beck depression inventory",
        "bdi",
        "bdi-ii",
        "beck depression inventory ii"
    ],
    
    # GDS相关
    "gds": [
        "geriatric depression scale",
        "gds",
        "gds-15",
        "gds-30"
    ],

    "hama": [
        "hamilton anxiety scale",
        "hama",
        "ham-a",
        "hamilton anxiety rating scale",
        "hamilton rating scale for anxiety"
    ]
}

# ==================== 帕金森病相关指标 ====================
PARKINSONS_OUTCOMES = {
    # UPDRS相关 - 添加更多子量表
    "updrs": [
        "unified parkinson disease rating scale",
        "unified parkinsons disease rating scale",
        "updrs",
        "updrs total",
        "updrs total score"
    ],
    
    "updrs-i": [
        "updrs part i",
        "updrs-i",
        "updrs part 1",
        "updrs mentation behavior and mood"
    ],
    
    "updrs-ii": [
        "updrs part ii",
        "updrs-ii", 
        "updrs part 2",
        "updrs activities of daily living"
    ],
    
    "updrs-iii": [
        "updrs part iii",
        "updrs-iii",
        "updrs part 3", 
        "motor updrs",
        "updrs motor examination",
        "updrs motor score"
    ],
    
    "updrs-iv": [
        "updrs part iv",
        "updrs-iv",
        "updrs part 4",
        "updrs complications"
    ],
    
    # PDQ相关
    "pdq": [
        "parkinson disease questionnaire",
        "parkinsons disease questionnaire",
        "pdq",
        "pdq-39",
        "pdq-8"
    ],
    
    # H&Y相关
    "hoehn-yahr": [
        "hoehn and yahr",
        "hoehn yahr",
        "h&y",
        "hy",
        "hoehn-yahr stage"
    ],
    
    # Schwab & England相关
    "schwab-england": [
        "schwab and england",
        "schwab england",
        "activities of daily living scale",
        "se-adl"
    ]
}

# ==================== 运动功能相关指标 ====================
MOTOR_OUTCOMES = {
    # Berg Balance Scale
    "bbs": [
        "berg balance scale",
        "bbs",
        "berg balance"
    ],
    
    # Timed Up and Go
    "tug": [
        "timed up and go",
        "tug",
        "time up and go"
    ],
    
    # 6分钟步行测试
    "6mwt": [
        "6 minute walk test",
        "6-minute walk test",
        "6mwt",
        "six minute walk test"
    ],
    "10mwt": [
        "10-meter walk test",
        "10 meter walk test",
        "10 metre walk test",
        "10-m walk test",
        "10 m walk test",
        "10mwt",
        "10 mwt",
        "10m-wt",
        "timed 10-meter walk test",
        "10-meter walking test",
        "10 meter walking test"
    ],
    "icars": [
        "international cooperative ataxia rating scale",
        "international cooperative ataxia rating scales",
        "icars",
        "icar",
        "rmicars",
        "rescaled modified international cooperative ataxia rating scale",
        "modified icars",
        "micars",
        "cagrs",
        "cooperative ataxia group rating scale",
        "icars-ocm",
        "international cooperative ataxia rating scale oculomotor",
        "icars weighted subscores",
        "icars subcomponent scores"
    ]
}

CEREBELLAR_ATAXIA_OUTCOMES={
    "sara": [
        "sara",
        "Scale for Assessment and Rating of Ataxia (SARA)",
        "SARA (Scale for the assessment and rating of ataxia)",
        "SARA total score",
        "Scale for Assessment and Rating of Ataxia (SARA) total score"
    ]
}

# ==================== 生活质量相关指标 ====================
QUALITY_OF_LIFE_OUTCOMES = {
    # SF-36相关
    "sf-36": [
        "short form 36",
        "sf-36",
        "sf36",
        "short form health survey"
    ],
    
    # EQ-5D相关
    "eq-5d": [
        "euroqol 5 dimension",
        "eq-5d",
        "eq5d",
        "euroqol"
    ]
}

# ==================== 脑区映射 ====================
BRAIN_REGIONS = {
    "dlpfc": [
        "dlpfc",
        "dorsolateral prefrontal",
        "dorsolateral prefrontal cortex",
        "prefrontal cortex",
        "brodmann area 8",
        "brodmann area 9"
    ],
    
    "motor cortex": [
        "motor cortex",
        "motor",
        "m1",
        "primary motor",
        "primary motor cortex",
        "motor area"
    ],
    
    "parietal": [
        "parietal",
        "parietal cortex",
        "angular gyrus",
        "ag",
        "precuneus"
    ],
    
    "temporal": [
        "temporal",
        "temporal cortex",
        "superior temporal",
        "superior temporal gyrus",
        "stg"
    ],
    
    "ifg": [
        "ifg",
        "inferior frontal gyrus",
        "inferior frontal"
    ],
    
    "vertex": [
        "vertex",
        "cz"
    ]
}

# ==================== 疾病/人群映射 ====================
DISEASE_CONDITIONS = {
    "alzheimer": [
        "alzheimer",
        "alzheimer's",
        "alzheimer's disease",
        "dementia",
        "alzheimer’s disease",
        "ad patients",
        "alzheimer disease"
    ],

    "mci": [
        "mci",
        "amci",
        "mild cognitive impairment",
        "mild cognitive impairment (mci)",
        "amnestic mild cognitive impairment"
    ],

    "depression": [
        "depression",
        "depressive",
        "depressive disorder",
        "mdd",
        "major depressive disorder",
        "major depression",
        "depressed patients"
    ],

    "treatment-resistant depression": [
        "treatment-resistant depression",
        "treatment resistant depression",
        "trd",
        "refractory depression",
        "treatment-refractory depression",
        "resistant depression",
        "difficult-to-treat depression",
        "trd patients"
    ],

    "parkinson": [
        "parkinson",
        "parkinson's",
        "parkinson's disease",
        "pd",
        "parkinsonian",
        "parkinsonism"
    ],
    
    "stroke": [
        "stroke",
        "cerebrovascular",
        "cerebrovascular accident",
        "cva",
        "cerebral infarction",
        "hemiparesis",
        "hemiplegia"
    ],

    "psci":[
        "psci",
        "post-stroke cognitive impairment",
        "post stroke cognitive impairment",
        "post-stroke cognitive impairment (psci)"
    ],
    
    "cerebellar ataxia": [
        "cerebellar ataxia",
        "ataxia",
        "ataxic",
        "cerebellar",
        "spinocerebellar ataxia",
        "sca",
        "friedreich ataxia",
        "friedreich's ataxia",
        "cerebellar degeneration",
        "cerebellar disorder"
    ],
    
    "anxiety": [
        "anxiety",
        "anxious",
        "anxiety disorder",
        "gad",
        "generalized anxiety",
        "generalized anxiety disorder"
    ],
    
    "schizophrenia": [
        "schizophrenia",
        "schizophrenic",
        "psychosis",
        "psychotic"
    ],
    
    "bipolar": [
        "bipolar",
        "bipolar disorder",
        "manic",
        "mania",
        "manic depression"
    ],
    
    "adhd": [
        "adhd",
        "attention deficit",
        "attention deficit hyperactivity disorder",
        "hyperactivity",
        "add"
    ],
    
    "autism": [
        "autism",
        "asd",
        "autistic",
        "autism spectrum",
        "autism spectrum disorder"
    ],
    
    "ocd": [
        "ocd",
        "obsessive",
        "compulsive",
        "obsessive-compulsive",
        "obsessive-compulsive disorder",
        "obsessive-compulsive disorder (ocd)",
        "obsessive compulsive disorder"
    ],

    "central stroke pain": [
        "central poststroke pain",
        "central post-stroke pain",
        "central post stroke pain",
        "cpsp",
        "thalamic pain syndrome",
        "dejerine-roussy syndrome",
        "central pain syndrome",
        "thalamic post-stroke pain syndrome",
        "centralized poststroke pain",
        "central neuropathic pain poststroke",
        "cva-related central pain",
        "poststroke central pain"
    ],

    "cdh": [
        "chronic daily headache",
        "cdh",
        "chronic daily headache syndrome",
        "transformed chronic daily headache",
        "t-cdh",
        "chronic daily headaches",
        "chronic headache",
        "chronic headaches",
        "daily chronic headache",
        "daily chronic headaches",
        "chronic daily head pain",
        "chronic daily cephalalgia"
    ],
    "chronic migraine": [
        "chronic migraine",
        "transformed migraine",
        "evolved migraine",
        "cm",
        "chronic migraine headache",
        "chronic migraine disorder",
        "migraine chronification",
        "chronic migraine syndrome",
        "transformed chronic migraine",
        "chronic migraine with medication overuse",
        "chronic migraine without aura",
        "chronic tension-type headache"
        "chronic migraine with aura",
        "refractory chronic migraine",
        "intractable chronic migraine",
        "high-frequency episodic migraine progression"
    ],
    "use disorder": [
        "substance abuse",
        "use disorder"
        "substance dependence",
        "drug dependence",
        "drug addiction",
        "substance use disorder",
        "heroin dependence",
        "heroin addiction",
        "drug addicts",
        "cocaine use disorder (cud)",
        "methamphetamine dependence",
        "methamphetamine addiction",
        "cocaine dependence",
        "cocaine addiction",
        "alcohol dependence",
        "alcohol addiction",
        "opioid dependence",
        "opioid addiction"
    ]
}

# ==================== 干预方法映射 ====================
# 层次化映射：父类 -> 子类列表
INTERVENTION_HIERARCHY = {
    # TMS大类
    "tms": [
        "tms",
        "transcranial magnetic stimulation",
        "rtms",
        "repetitive tms",
        "tbs",
        "theta burst stimulation",
        "single pulse tms",
        "peripheral magnetic stimulation",
        "dtms",
        "atms",
        "artms"
    ],
    
    # rTMS类
    "rtms": [
        "rtms",
        "repetitive tms",
        "repetitive transcranial magnetic stimulation",
        "tbs",
        "theta burst stimulation",
        "ctbs",
        "continuous theta burst",
        "theta-burst stimulation",
        "itbs",
        "intermittent theta burst",
        "high frequency rtms",
        "low frequency rtms",
        "intermittent theta-burst stimulation",
        "accelerated intermittent theta-burst stimulation",
        "deep repetitive transcranial magnetic stimulation"
    ],
    
    # TBS类
    "tbs": [
        "tbs",
        "theta burst stimulation",
        "ctbs",
        "continuous theta burst",
        "itbs",
        "intermittent theta burst"
    ],
    
    # tDCS类
    "tdcs": [
        "tdcs",
        "transcranial direct current stimulation",
        "anodal tdcs",
        "cathodal tdcs"
    ],
    
    # DBS类
    "dbs": [
        "dbs",
        "deep brain stimulation"
    ],

    "dtms": ["dtms", "deep tms", "deep transcranial magnetic stimulation","deep transcranial magnetic stimulation (dtms)"],

    # 具体子类
    "ctbs": ["ctbs", "continuous theta burst"],
    "itbs": ["itbs", "intermittent theta burst"],
    "high frequency": ["high frequency", "high-frequency", "high frequency rtms"],
    "low frequency": ["low frequency", "low-frequency", "low frequency rtms"],

    "atms": ["atms","atbs","artms","actbs",
             "accelerated repetitive transcranial magnetic stimulation","accelerated intermittent theta burst stimulation",
             "accelerated theta burst stimulation","accelerated continuous theta burst stimulation",
             "accelerated high-dose theta burst stimulation","accelerated continuous theta-burst stimulation","accelerated ctbs",
             "accelerated deep transcranial magnetic stimulation"]
}

# 同义词映射（同级别的不同表达）
INTERVENTION_SYNONYMS = {
    # rTMS相关同义词
    "rtms": ["rtms", "repetitive tms", "repetitive transcranial magnetic stimulation", "r-tms"],
    "repetitive tms": ["rtms", "repetitive tms", "repetitive transcranial magnetic stimulation", "r-tms"],
    "repetitive transcranial magnetic stimulation": ["rtms", "repetitive tms", "repetitive transcranial magnetic stimulation", "r-tms"],
    
    # TMS相关同义词
    "tms": ["tms", "transcranial magnetic stimulation"],
    "transcranial magnetic stimulation": ["tms", "transcranial magnetic stimulation"],
    
    # TBS相关同义词
    "tbs": ["tbs", "theta burst stimulation", "theta-burst stimulation"],
    "theta burst stimulation": ["tbs", "theta burst stimulation", "theta-burst stimulation"],
    "ctbs": ["ctbs", "continuous theta burst", "continuous theta burst stimulation", "continuous tbs"],
    "continuous theta burst": ["ctbs", "continuous theta burst", "continuous theta burst stimulation", "continuous tbs"],
    "itbs": ["itbs", "intermittent theta burst", "intermittent theta burst stimulation", "intermittent tbs"],
    "intermittent theta burst": ["itbs", "intermittent theta burst", "intermittent theta burst stimulation", "intermittent tbs"],
    
    # tDCS相关同义词
    "tdcs": ["tdcs", "transcranial direct current stimulation", "t-dcs"],
    "transcranial direct current stimulation": ["tdcs", "transcranial direct current stimulation", "t-dcs"],
    "anodal tdcs": ["anodal tdcs", "anodal stimulation", "anodal tDCS"],
    "cathodal tdcs": ["cathodal tdcs", "cathodal stimulation", "cathodal tDCS"],
    
    # DBS相关同义词
    "dbs": ["dbs", "deep brain stimulation"],
    "deep brain stimulation": ["dbs", "deep brain stimulation"],

    "dtms": ["dtms", "deep tms", "deep transcranial magnetic stimulation"],
    "drtms": ["deep repetitive transcranial magnetic stimulation", "drtms", "deep rtms"],
    
    # 频率相关同义词
    "high frequency": ["high frequency", "high-frequency", "hf", "high freq"],
    "low frequency": ["low frequency", "low-frequency", "lf", "low freq"],

    "atms": ["atms", "accelerated transcranial magnetic stimulation", "accelerated tms"],
    "artms":["artms", "accelerated repetitive transcranial magnetic stimulation", "accelerated rtms",
             "accelerated repetitive tms", "accelerated rtms protocol", "accelerated rtms (artms)"],
    "aitbs":["aitbs","accelerated intermittent theta burst stimulation", "accelerated intermittent theta burst",
             "accelerated intermittent tbs", "accelerated intermittent theta-burst stimulation"],
    "actbs":["actbs", "accelerated continuous theta-burst", "accelerated continuous theta burst stimulation"],
    "atbs": ["atbs", "accelerated theta burst stimulation", "accelerated theta-burst stimulation"],

    "fmri": ["fmri", "functional magnetic resonance imaging", "functional mri"],

    "cognitive training":["cognitive rehabilitation training", "cognitive training", "cognitive training (ct)", "cognitive rehabilitation","neurorehabilitation"],
    "cognitive impairment":["cognitive impairment","cognitive improvement","cognitive function","neurocognition",
                            "cognitive effects","cognitive adverse effects","cognitive flexibility","cognitive dysfunction"],
    "non-pharmaceutical therapies":["audio-guided mindfulness meditation","guided meditations","behavioral activation (ba) therapy",
                                    "cognitive training","bright light therapy","emotional directives","treadmill walking","trauma script exposure",
                                    "mindfulness-based stress reduction","pe therapy","aerobic exercise (aex)",
                                    "structured cognitive behavioral therapy (cbt)","exposure therapy","cognitive behavioral therapy (cbt)",
                                    "non-pharmaceutical therapies","behavioral activation therapy","simultaneous psychotherapy","computerized cognitive training"],

    "drug therapy":["fluoxetine","drug therapy","sertraline"],
    "peripheral magnetic stimulation": ["peripheral magnetic stimulation","peripheral repetitive magnetic stimulation",
                                        "rpms","peripheral repetitive magnetic stimulation (rpms)","pms"]

}

# ==================== 研究设计类型 ====================
STUDY_DESIGNS = {
    "rct": [
        "rct",
        "randomized controlled trial",
        "randomized",
        "randomised controlled trial",
        "randomised",
        "randomized, double-blind, sham-controlled trial"
    ],
    
    "double-blind": [
        "double-blind",
        "double blind",
        "double-blinded",
        "double blinded"
    ],
    
    "single-blind": [
        "single-blind",
        "single blind",
        "single-blinded",
        "single blinded"
    ],
    
    "crossover": [
        "crossover",
        "cross-over",
        "cross over"
    ],
    
    "parallel": [
        "parallel",
        "parallel group",
        "parallel-group"
    ],
    "open-label":[
        "open-label",
        "open label",
        "open-label study",
        "open trial"
    ]
}

# ==================== 查询概念提取映射 ====================
# 用于从用户查询中提取核心概念，复用已有的映射

# 干预方法查询关键词（基于 INTERVENTION_SYNONYMS）
INTERVENTION_QUERY_KEYWORDS = {
    "rtms": INTERVENTION_SYNONYMS["rtms"],
    "tms": INTERVENTION_SYNONYMS["tms"],
    "tdcs": INTERVENTION_SYNONYMS["tdcs"],
    "dbs": INTERVENTION_SYNONYMS["dbs"],
    "tbs": INTERVENTION_SYNONYMS["tbs"],
    "itbs": INTERVENTION_SYNONYMS["itbs"],
    "ctbs": INTERVENTION_SYNONYMS["ctbs"],
    "high frequency": INTERVENTION_SYNONYMS["high frequency"],
    "low frequency": INTERVENTION_SYNONYMS["low frequency"],
    "atms": INTERVENTION_SYNONYMS["atms"]
}


# ==================== 合并所有指标 ====================
OUTCOME_ALIASES = {
    **COGNITIVE_OUTCOMES,
    **DEPRESSION_OUTCOMES,
    **PARKINSONS_OUTCOMES,
    **MOTOR_OUTCOMES,
    **QUALITY_OF_LIFE_OUTCOMES,
    **OCD_OUTCOMES,
    **CEREBELLAR_ATAXIA_OUTCOMES
}




# ==================== 按疾病分类的指标组 ====================
DISEASE_SPECIFIC_OUTCOMES = {
    "alzheimer": COGNITIVE_OUTCOMES,
    "dementia": COGNITIVE_OUTCOMES,
    "depression": DEPRESSION_OUTCOMES,
    "parkinsons": PARKINSONS_OUTCOMES,
    "parkinson": PARKINSONS_OUTCOMES
}