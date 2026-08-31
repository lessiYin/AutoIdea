const ACTIVE_STATUSES = new Set(["queued", "running", "waiting_for_input"]);
const FILTER_VIEWS = new Set(["map", "papers", "evidence", "ideas", "artifacts"]);
const TERMINAL_STATUSES = new Set(["pipeline_completed", "failed", "stopped", "stale", "checkpoint_reached"]);

const state = {
  snapshot: null,
  snapshotSignature: "",
  rootSnapshot: null,
  runs: [],
  runsSignature: "",
  selectedRunId: initialParam("run"),
  activeView: initialParam("view") || "studio",
  language: getInitialLanguage(),
  query: initialParam("q").trim().toLocaleLowerCase(),
  confidence: initialParam("confidence"),
  nodeKind: initialParam("node"),
  selectedGraphNode: null,
  focusNode: null,
  config: null,
  configDraft: {},
  activeConfigGroup: initialParam("group") || "quick",
  configStatus: "",
  configStatusKind: "",
  globalStatus: { key: "", args: [], detail: "" },
  configDirty: new Set(),
  interactionDraft: {},
  artifactCache: new Map(),
  pollTimer: null,
  activityClockTimer: null,
  logTimer: null,
  activeLogRunId: null,
  dialogReturnFocus: null,
  toastTimer: null,
  graph: { frame: null, transform: { x: 0, y: 0, k: 1 } },
  navOpen: false,
  runSelectorMarkup: "",
  studioDraftInitialized: false,
  studioDraft: {
    prompt: "",
    runName: "",
    workspace: "",
    model: "",
    provider: "",
    seedPapers: "",
    seedIdeas: "",
    autoApprove: true,
    showThinking: true,
    mode: "new",
  },
};

const refs = {
  main: document.querySelector("#mainContent"),
  nav: document.querySelector("#primaryNav"),
  navToggle: document.querySelector("#navToggle"),
  navScrim: document.querySelector("#navScrim"),
  viewRoot: document.querySelector("#viewRoot"),
  status: document.querySelector("#status"),
  workspaceName: document.querySelector("#workspaceName"),
  runSelector: document.querySelector("#runSelector"),
  runStatus: document.querySelector("#runStatusPill"),
  liveBadge: document.querySelector("#liveRunBadge"),
  languageToggle: document.querySelector("#languageToggle"),
  refreshButton: document.querySelector("#refreshButton"),
  newRunButton: document.querySelector("#newRunButton"),
  pageHeader: document.querySelector("#pageHeader"),
  pageEyebrow: document.querySelector("#pageEyebrow"),
  pageTitle: document.querySelector("#pageTitle"),
  pageDescription: document.querySelector("#pageDescription"),
  pageActions: document.querySelector("#pageActions"),
  contextTools: document.querySelector("#contextTools"),
  searchInput: document.querySelector("#searchInput"),
  confidenceControl: document.querySelector("#confidenceControl"),
  confidenceFilter: document.querySelector("#confidenceFilter"),
  nodeControl: document.querySelector("#nodeControl"),
  nodeFilter: document.querySelector("#nodeFilter"),
  artifactDialog: document.querySelector("#artifactDialog"),
  artifactTitle: document.querySelector("#artifactTitle"),
  artifactPath: document.querySelector("#artifactPath"),
  artifactBody: document.querySelector("#artifactBody"),
  closeArtifactButton: document.querySelector("#closeArtifactButton"),
  toast: document.querySelector("#toastRegion"),
};

const I18N = {
  en: {
    observatory: "Research Observatory", runContext: "Research run", selectRun: "Select research run",
    rootWorkspace: "Workspace overview", newRun: "New research", observe: "Observe", research: "Research",
    inspect: "Inspect", navOverview: "Overview", navLive: "Live run", navReport: "Final report",
    navMap: "Literature map", navPapers: "Papers", navEvidence: "Evidence", navIdeas: "Ideas",
    navPipeline: "Pipeline", navArtifacts: "Artifacts", navSettings: "Settings", localOnly: "Local workspace",
    search: "Search", searchPlaceholder: "Search this view…", confidence: "Confidence", nodeType: "Node type",
    all: "All", high: "High", medium: "Medium", low: "Low", papers: "Papers", claims: "Claims",
    ideas: "Ideas", gaps: "Gaps", artifacts: "Artifacts", artifact: "Artifact", close: "Close",
    loading: "Loading the selected research workspace…", loaded: "Research workspace is current.",
    refresh: "Refresh research data", switchLanguage: "Switch language", openNavigation: "Open navigation",
    closeNavigation: "Close navigation", noRun: "No run selected", workspace: "Workspace",
    status: "Status", stage: "Stage", progress: "Progress", checkpoints: "Checkpoint decisions",
    overviewEyebrow: "Evidence to decision", overviewTitle: "Observe the full research trace.",
    overviewDescription: "Start, supervise, and verify every step from a research question to an evidence-grounded final report.",
    heroEyebrow: "AUTOIDEA", heroTitle: "Literature evidence synthesis and research direction assessment",
    heroBody: "Run and inspect the complete workflow from literature retrieval and evidence binding through research-gap identification, candidate-idea review, feasibility assessment, and final report generation.",
    startResearch: "Start research", observeRun: "Observe selected run", noActiveRun: "Select or start a run to see its live research trace.",
    traceTitle: "Research Trace", traceAwaiting: "Awaiting a new question", traceComplete: "Research record verified",
    traceRunning: "Pipeline in motion", traceWaiting: "Human decision required", stages: "Stages", complete: "Complete",
    pending: "Pending", checkpoint: "Human checkpoint", recentRuns: "Recent research", recentRunsBody: "Every run keeps its own workspace, thread, settings, events, and completion proof.",
    noRuns: "No browser-managed runs yet.", createFirst: "Create the first research run to begin an inspectable record.",
    composeTitle: "Frame a research question", composeBody: "A clear question and constraints become the source record for the entire pipeline.",
    runMode: "Run mode", modeNew: "New run", modeResume: "Resume selected", modeFollowup: "Follow-up",
    researchQuestion: "Research question", promptPlaceholder: "Describe the domain, unresolved problem, constraints, and the kind of contribution you want to explore…",
    runName: "Run name", runNamePlaceholder: "e.g. long-video-agents", workspaceOverride: "New workspace location (optional)",
    workspacePlaceholder: "Leave blank for runs/<unique-name>", advancedOptions: "Model, seeds & execution options",
    provider: "Provider", model: "Model", configuredDefault: "Use effective default", seedPapers: "Seed papers JSON",
    seedIdeas: "Seed ideas file", autoApprove: "Fully automatic (default)", autoApproveHelp: "Checked: Stage 7, 9, and 10 are recorded and approved automatically. Unchecked: the run pauses at those three checkpoints for your review.",
    runOverridesHelp: "Provider and model entered here apply to this run and override environment and saved defaults.",
    showThinking: "Keep detailed thinking in the local log", launchRun: "Start research run", resumeRun: "Resume selected run",
    followupRun: "Start follow-up", selectedContext: "The selected run's workspace and thread are reused. Its model and seed values are prefilled below and may be overridden for this run.",
    liveEyebrow: "Run observatory", liveTitle: "Supervise the active research process.", liveDescription: "Respond to structured questions, inspect stage evidence, and intervene without parsing terminal output.",
    chooseRun: "Choose a research run", chooseRunBody: "Use the run selector above or start a new run. Results and artifacts always follow that selection.",
    runDetail: "Run detail", thread: "Thread", process: "Process", started: "Started", finished: "Finished",
    inputRequired: "Your decision is required", inputBody: "The agent is paused. Your response will resume this exact LangGraph thread.",
    toolApproval: "Tool approval", toolApprovalBody: "Review the requested local action before it runs.", approve: "Approve",
    reject: "Reject", submitDecision: "Submit decision", otherAnswer: "Other answer", optional: "Optional",
    log: "Research log", logBody: "Raw process output is secondary evidence; structured state above remains authoritative.", openFullLog: "Open full log",
    stopRun: "Stop run", reportEyebrow: "Verified output", reportTitle: "Read the final research report.",
    reportDescription: "Stage 12 is shown as complete only when the final report, all artifacts, all three human decisions, and the audit agree.",
    finalReport: "Final report", finalNotReady: "The final report is not ready.", finalNotReadyBody: "Continue the selected run through Stage 12. This view will update automatically when final_report.md is verified.",
    supportingArtifacts: "Supporting record", topIdeas: "Top candidate ideas", noIdeas: "No research ideas are available in this run yet.",
    mapEyebrow: "Evidence topology", mapTitle: "Trace papers through evidence and gaps into research ideas.", mapDescription: "Only structured relationships recorded in the selected run are drawn; the table below is the accessible source of truth.",
    papersEyebrow: "Literature corpus", papersTitle: "Inspect the papers behind the synthesis.", papersDescription: "Papers selected into the evidence ledger stay connected to their claims and downstream ideas.",
    evidenceEyebrow: "Claim ledger", evidenceTitle: "Audit every evidence claim.", evidenceDescription: "Confidence, source identity, and citation provenance remain visible together.",
    ideasEyebrow: "Candidate directions", ideasTitle: "Compare ideas against their evidence.", ideasDescription: "Scores are context, while gaps, mechanisms, and cited claims provide the research argument.",
    pipelineEyebrow: "Completion proof", pipelineTitle: "Verify the complete AutoIdea pipeline.", pipelineDescription: "Observed files override stale self-reported state; a process exit alone never means completion.",
    artifactsEyebrow: "Research record", artifactsTitle: "Open every generated artifact.", artifactsDescription: "Artifacts are scoped to the selected run so historical files cannot leak into current results.",
    settingsEyebrow: "Local configuration", settingsTitle: "Configure the research environment.", settingsDescription: "Secrets remain masked and blank secret fields preserve their existing values.",
    completionProof: "Completion proof", allArtifacts: "All stage artifacts", finalReportProof: "Stage 12 final report", gateProof: "Stage 12 gate", checkpointProof: "Three checkpoint decisions", auditProof: "Artifact integrity audit",
    verified: "Verified", notVerified: "Not verified", notChecked: "Not checked", missing: "Missing", recorded: "Recorded",
    pipelineStages: "Pipeline stages", requiredArtifacts: "Required artifacts", checkpointRecorded: "Decision recorded",
    noMatches: "No items match the current filters.", noArtifacts: "No artifacts are present in this selected workspace.",
    mapInspector: "Map inspector", mapHint: "Select a node to inspect its provenance and visible relationships.",
    focusNeighborhood: "Focus neighbors", clearFocus: "Clear focus", zoomIn: "Zoom in", zoomOut: "Zoom out", resetView: "Reset view",
    accessibleRelationships: "Accessible relationship list", connections: "Connections", noConnections: "No visible connections.",
    title: "Title", source: "Source", year: "Year", relevance: "Relevance", weakestLink: "Weakest link",
    citation: "Citation", claim: "Claim", section: "Section", score: "Score", targetGaps: "Target gaps", evidenceLinks: "Evidence links",
    openArtifact: "Open artifact", openReport: "Open final report", viewRun: "View run", select: "Select", actions: "Actions",
    saveSettings: "Save settings", reloadSettings: "Reload", configFile: "Configuration file", currentEffective: "Effective default",
    secretHelp: "Leave blank to preserve the current secret.", envOverride: "Environment override", saving: "Saving settings…",
    saved: "Defaults saved.", savedWithOverrides: (names) => `Defaults saved, but ${names} still override the changed fields.`, answerAccepted: "Response accepted. The research process is resuming.", runStarted: "Research run started.",
    runStopped: "Research run stopped.", loadingReport: "Loading final report…", staleState: "Persisted pipeline state differs from observed artifacts.",
    selectGraphNode: "Select graph node", showingNodes: (n, e) => `${n} nodes · ${e} relationships shown`,
    countPapers: "Papers", countClaims: "Evidence claims", countIdeas: "Candidate ideas", countArtifacts: "Artifacts", countProgress: "Pipeline complete",
    errorPrefix: "Action could not be completed", retry: "Retry", exitCode: "Exit code", viewLog: "View log",
  },
  zh: {
    observatory: "研究观测台", runContext: "研究运行", selectRun: "选择研究运行", rootWorkspace: "工作区总览",
    newRun: "新建研究", observe: "观测", research: "研究", inspect: "检查", navOverview: "总览", navLive: "实时运行",
    navReport: "最终报告", navMap: "文献图谱", navPapers: "论文", navEvidence: "证据", navIdeas: "想法",
    navPipeline: "流程", navArtifacts: "产物", navSettings: "设置", localOnly: "本地工作区", search: "搜索",
    searchPlaceholder: "搜索当前视图…", confidence: "置信度", nodeType: "节点类型", all: "全部", high: "高",
    medium: "中", low: "低", papers: "论文", claims: "证据", ideas: "想法", gaps: "研究空白", artifacts: "产物",
    artifact: "产物", close: "关闭", loading: "正在加载所选研究工作区…", loaded: "研究工作区已是最新。",
    refresh: "刷新研究数据", switchLanguage: "切换语言", openNavigation: "打开导航", closeNavigation: "关闭导航",
    noRun: "尚未选择运行", workspace: "工作区", status: "状态", stage: "阶段", progress: "进度", checkpoints: "人工决策",
    overviewEyebrow: "从证据到决策", overviewTitle: "观测完整的研究轨迹。", overviewDescription: "从研究问题到有证据支撑的最终报告，启动、监督并验证每一个步骤。",
    heroEyebrow: "AUTOIDEA", heroTitle: "文献证据综合与研究方向评估", heroBody: "用于运行和检查从文献检索、证据绑定、研究空白识别，到候选想法评审、可行性分析与最终报告生成的完整流程。",
    startResearch: "开始研究", observeRun: "观测所选运行", noActiveRun: "选择或启动一个运行，即可看到实时研究轨迹。",
    traceTitle: "研究轨迹", traceAwaiting: "等待新的研究问题", traceComplete: "研究记录已验证", traceRunning: "流程正在推进",
    traceWaiting: "需要人工决策", stages: "阶段", complete: "已完成", pending: "待处理", checkpoint: "人工检查点",
    recentRuns: "最近研究", recentRunsBody: "每次运行都独立保存工作区、线程、设置、事件和完成证明。", noRuns: "尚无网页管理的运行。",
    createFirst: "创建第一个研究运行，开始一条可检查的研究记录。", composeTitle: "定义研究问题", composeBody: "清晰的问题与约束会成为整个流程的源记录。",
    runMode: "运行模式", modeNew: "新运行", modeResume: "恢复所选运行", modeFollowup: "后续研究", researchQuestion: "研究问题",
    promptPlaceholder: "描述领域、尚未解决的问题、约束，以及希望探索的研究贡献…", runName: "运行名称", runNamePlaceholder: "例如 long-video-agents",
    workspaceOverride: "新工作区位置（可选）", workspacePlaceholder: "留空则使用 runs/<唯一名称>", advancedOptions: "模型、种子与执行选项",
    provider: "供应商", model: "模型", configuredDefault: "使用当前有效默认值", seedPapers: "种子论文 JSON", seedIdeas: "种子想法文件",
    autoApprove: "全自动运行（默认）", autoApproveHelp: "勾选后，Stage 7、9、10 会自动记录并批准，无需回答；取消勾选后，流程会在这三个检查点等待人工审查。", showThinking: "在本地日志保留详细思考流",
    runOverridesHelp: "此处填写的供应商和模型仅用于本次运行，并覆盖环境变量和已保存默认值。",
    launchRun: "启动研究运行", resumeRun: "恢复所选运行", followupRun: "启动后续研究", selectedContext: "将复用所选运行的工作区和线程；下方会预填其模型与种子配置，也可为本次运行覆盖。",
    liveEyebrow: "运行观测", liveTitle: "监督正在执行的研究流程。", liveDescription: "回答结构化问题、检查阶段证据并进行干预，无需解析终端文本。",
    chooseRun: "选择一个研究运行", chooseRunBody: "使用上方运行选择器或新建运行。结果和产物始终跟随该选择。", runDetail: "运行详情",
    thread: "线程", process: "进程", started: "开始时间", finished: "结束时间", inputRequired: "需要你的决策",
    inputBody: "智能体已暂停；提交回答后会恢复同一个 LangGraph 线程。", toolApproval: "工具审批", toolApprovalBody: "请先检查本地操作，再决定是否执行。",
    approve: "批准", reject: "拒绝", submitDecision: "提交决策", otherAnswer: "其他回答", optional: "可选", log: "研究日志",
    logBody: "原始进程输出是辅助证据；上方结构化状态才是权威来源。", openFullLog: "打开完整日志", stopRun: "停止运行",
    reportEyebrow: "已验证输出", reportTitle: "阅读最终研究报告。", reportDescription: "只有最终报告、全部产物、三次人工决策和审计结果一致时，Stage 12 才显示完成。",
    finalReport: "最终报告", finalNotReady: "最终报告尚未就绪。", finalNotReadyBody: "继续所选运行直至 Stage 12；final_report.md 验证完成后，本页会自动更新。",
    supportingArtifacts: "支撑记录", topIdeas: "候选想法", noIdeas: "本次运行尚未生成研究想法。", mapEyebrow: "证据拓扑",
    mapTitle: "追溯论文如何经由证据与研究空白形成研究想法。", mapDescription: "只绘制所选运行中明确记录的结构化关系；下方关系列表是可访问的事实底稿。",
    papersEyebrow: "文献语料", papersTitle: "检查综合结论背后的论文。", papersDescription: "进入证据账本的论文会与其论据和下游想法保持连接。",
    evidenceEyebrow: "论据账本", evidenceTitle: "审计每一条证据。", evidenceDescription: "置信度、来源身份和引用出处同时可见。",
    ideasEyebrow: "候选方向", ideasTitle: "对照证据比较研究想法。", ideasDescription: "分数只提供背景，研究空白、机制和引用证据才构成论证。",
    pipelineEyebrow: "完成证明", pipelineTitle: "验证完整的 AutoIdea 流程。", pipelineDescription: "以实际文件覆盖过时的自报状态；进程退出本身绝不等于完成。",
    artifactsEyebrow: "研究记录", artifactsTitle: "打开每一项生成产物。", artifactsDescription: "产物严格限定到所选运行，历史文件不会混入当前结果。",
    settingsEyebrow: "本地配置", settingsTitle: "配置研究环境。", settingsDescription: "密钥始终遮蔽；密钥字段留空会保留现有值。",
    completionProof: "完成证明", allArtifacts: "全部阶段产物", finalReportProof: "Stage 12 最终报告", gateProof: "Stage 12 门禁", checkpointProof: "三次检查点决策", auditProof: "产物完整性审计",
    verified: "已验证", notVerified: "未验证", notChecked: "未检查", missing: "缺失", recorded: "已记录", pipelineStages: "流程阶段",
    requiredArtifacts: "所需产物", checkpointRecorded: "决策已记录", noMatches: "没有符合当前筛选条件的内容。", noArtifacts: "所选工作区尚无产物。",
    mapInspector: "图谱检查器", mapHint: "选择节点，检查其来源和可见关系。", focusNeighborhood: "聚焦相邻节点", clearFocus: "清除聚焦",
    zoomIn: "放大", zoomOut: "缩小", resetView: "重置视图", accessibleRelationships: "可访问关系列表", connections: "连接", noConnections: "没有可见连接。",
    title: "标题", source: "来源", year: "年份", relevance: "相关性", weakestLink: "最弱环节", citation: "引用", claim: "论据", section: "章节",
    score: "评分", targetGaps: "目标空白", evidenceLinks: "证据连接", openArtifact: "打开产物", openReport: "打开最终报告", viewRun: "查看运行",
    select: "选择", actions: "操作", saveSettings: "保存设置", reloadSettings: "重新加载", configFile: "配置文件", currentEffective: "当前有效默认值",
    secretHelp: "留空会保留当前密钥。", envOverride: "环境变量覆盖", saving: "正在保存设置…", saved: "默认配置已保存。",
    savedWithOverrides: (names) => `默认配置已保存，但 ${names} 仍在覆盖刚修改的字段。`,
    answerAccepted: "回答已接收，研究流程正在恢复。", runStarted: "研究运行已启动。", runStopped: "研究运行已停止。", loadingReport: "正在加载最终报告…",
    staleState: "持久化流程状态与实际产物不一致。", selectGraphNode: "选择图谱节点", showingNodes: (n, e) => `显示 ${n} 个节点 · ${e} 条关系`,
    countPapers: "论文", countClaims: "证据论据", countIdeas: "候选想法", countArtifacts: "产物", countProgress: "流程完成度",
    errorPrefix: "操作未能完成", retry: "重试", exitCode: "退出码", viewLog: "查看日志",
  },
};

Object.assign(I18N.en, {
  process: "Process",
  queued: "Queued",
  running: "Running",
  waitingForInput: "Waiting for input",
  pipelineCompleted: "Pipeline complete",
  failed: "Failed",
  stopped: "Stopped",
  stale: "Stale",
  checkpointReached: "Checkpoint reached",
  notStarted: "Not started",
  reportReady: "The complete, audited research report is ready.",
  reportReading: "Rendered directly from the selected run's final_report.md.",
  paperCount: "paper corpus",
  noPapers: "No papers are available in this selected workspace.",
  noEvidence: "No evidence claims are available in this selected workspace.",
  authors: "Authors",
  venue: "Venue",
  position: "Critical position",
  initialAttack: "Initial attack",
  dimensions: "Review dimensions",
  description: "Description",
  selfAssessment: "Self-assessment",
  explored: "Explored",
  unexplored: "Unexplored",
  designSpace: "Design space",
  fileSize: "File size",
  fileType: "File type",
  noPreview: "This artifact has no rendered preview.",
  checkpointRequired: "Human decision required",
  checkpointNotRecorded: "Decision not recorded",
  artifactMissing: "Required artifact missing or invalid",
  artifactReady: "Artifact verified",
  auditIssues: "Audit issues",
  relationshipFrom: "From",
  relationshipTo: "To",
  relationshipType: "Relationship",
  selectedNode: "Selected node",
  graphInstructions: "Drag the canvas to pan, use the toolbar or wheel to zoom, and press Enter on a focused node to inspect it.",
  paperEvidenceMapping: "Paper → evidence mapping",
  evidenceGapMapping: "Evidence → gap mapping",
  provenanceComplete: "Every evidence claim resolves to a registered paper.",
  provenanceMissing: (count) => `${count} evidence claim${count === 1 ? "" : "s"} lack a resolvable paper source. No inferred edges were drawn.`,
  provenancePaperUse: (used, total, unused) => `${used}/${total} papers contribute evidence; ${unused} remain discovery-corpus papers without an evidence-ledger claim.`,
  gapProvenanceComplete: "Every registered research gap has structured Claim-level provenance.",
  gapProvenanceMissing: (count) => `${count} research gap${count === 1 ? "" : "s"} lack a valid evidence link. Unknown Claim IDs are never drawn.`,
  graphScopeTitle: "How are graph relationships validated?",
  graphScopeBody: "Paper → evidence comes from evidence_db.json, evidence → gap from research_gaps.json, and gap/evidence → idea from raw_ideas.json. Unknown IDs and invalid relationship types are rejected rather than inferred.",
  relationshipSupports: "supports evidence",
  relationshipSupportsGap: "supports gap",
  relationshipPartialCoverage: "partially covers gap",
  relationshipChallengesGap: "challenges gap",
  relationshipEvidenceFor: "supports idea",
  relationshipTargets: "motivates idea",
  relationshipBasis: "Recorded rationale",
  configIntro: "This page saves defaults for future runs. Environment variables override saved defaults; values entered when starting a run override both.",
  configured: "Configured",
  notConfigured: "Not configured",
  saveCurrentGroup: "Save current group",
  reloaded: "Settings reloaded from disk.",
  loadingArtifact: "Loading artifact…",
  loadingLog: "Loading run log…",
  logUnavailable: "No process output has been recorded for this run.",
  confirmStop: "Stop this local research process? Generated artifacts will be preserved.",
  unknown: "Unknown",
  currentActivity: "Current activity",
  stageProgress: "Stage progress",
  lastActivity: "Last activity",
  activityNow: "just now",
  activitySecondsAgo: (count) => `${count}s ago`,
  activityMinutesAgo: (count) => `${count}m ago`,
  activityHoursAgo: (count) => `${count}h ago`,
  activityDaysAgo: (count) => `${count}d ago`,
  phase_defining_requirements: "Defining research requirements",
  phase_formalizing_problem: "Formalizing the research problem",
  phase_surveying_literature: "Surveying literature",
  phase_reading_papers: "Reading selected papers",
  phase_positioning_papers: "Positioning papers",
  phase_expanding_literature: "Expanding the literature set",
  phase_binding_evidence: "Binding claims to sources",
  phase_synthesizing_gaps: "Synthesizing research gaps",
  phase_mapping_design_space: "Mapping the design space",
  phase_generating_ideas: "Generating candidate ideas",
  phase_ranking_ideas: "Ranking candidate ideas",
  phase_debating_ideas: "Running adversarial review",
  phase_assessing_feasibility: "Assessing feasibility",
  phase_writing_report: "Writing the final report",
  phase_preparing_batches: "Preparing work batches",
  phase_processing_batch: "Processing a work batch",
  phase_checking_batches: "Checking batch results",
  phase_merging_batches: "Merging batch results",
  phase_searching_sources: "Searching literature sources",
  phase_retrieving_full_text: "Retrieving paper full text",
  phase_validating_stage: "Validating the stage",
  phase_recording_reflection: "Recording stage reflection",
  phase_writing_artifact: "Writing a stage artifact",
  phase_running_subagent: "Running a research sub-agent",
  phase_integrating_subagent: "Integrating sub-agent results",
  phase_reasoning: "Analyzing research evidence",
  phase_runtime_error: "A runtime operation failed",
  unit_papers_collected: "papers collected",
  unit_papers_processed: "papers processed",
  unit_papers_positioned: "papers positioned",
  unit_batches: "batches",
  unit_ideas_generated: "ideas generated",
  unit_ideas_ranked: "ideas ranked",
  unit_ideas_reviewed: "ideas reviewed",
  unit_ideas_assessed: "ideas assessed",
  unit_stage: "stage complete",
  metric_full_text: "Full text",
  metric_failed: "Failed",
  metric_batches: "Batches",
  metric_claims: "Claims",
  metric_gaps: "Research gaps",
  metric_evidence_links: "Evidence links",
  metric_axes: "Design axes",
  metric_combinations: "Combinations",
  metric_comparisons: "Comparisons",
  metric_debate_rounds: "Debate rounds",
});

Object.assign(I18N.zh, {
  process: "进程",
  queued: "排队中",
  running: "运行中",
  waitingForInput: "等待输入",
  pipelineCompleted: "流程已完成",
  failed: "失败",
  stopped: "已停止",
  stale: "状态失联",
  checkpointReached: "已到达检查点",
  notStarted: "尚未开始",
  reportReady: "完整且通过审计的研究报告已经就绪。",
  reportReading: "内容直接来自所选运行的 final_report.md。",
  paperCount: "篇论文语料",
  noPapers: "所选工作区尚无论文。",
  noEvidence: "所选工作区尚无证据论据。",
  authors: "作者",
  venue: "发表场所",
  position: "批判性定位",
  initialAttack: "初始攻击点",
  dimensions: "评审维度",
  description: "说明",
  selfAssessment: "自评",
  explored: "已探索",
  unexplored: "未探索",
  designSpace: "设计空间",
  fileSize: "文件大小",
  fileType: "文件类型",
  noPreview: "该产物没有可渲染的预览。",
  checkpointRequired: "需要人工决策",
  checkpointNotRecorded: "尚未记录决策",
  artifactMissing: "所需产物缺失或无效",
  artifactReady: "产物已验证",
  auditIssues: "审计问题",
  relationshipFrom: "起点",
  relationshipTo: "终点",
  relationshipType: "关系",
  selectedNode: "所选节点",
  graphInstructions: "拖动画布可平移，使用工具栏或滚轮缩放，聚焦节点后按 Enter 检查。",
  paperEvidenceMapping: "论文 → 证据映射",
  evidenceGapMapping: "证据 → Research Gap 映射",
  provenanceComplete: "所有证据都能回溯到论文登记表。",
  provenanceMissing: (count) => `有 ${count} 条证据无法解析论文来源；系统没有为它们猜测或伪造连线。`,
  provenancePaperUse: (used, total, unused) => `证据库使用了 ${used}/${total} 篇论文；其余 ${unused} 篇仅属于检索语料，尚未贡献入库证据。`,
  gapProvenanceComplete: "每个已登记的 Research Gap 都有结构化的 Claim 级依据。",
  gapProvenanceMissing: (count) => `有 ${count} 个 Research Gap 缺少有效证据映射；未知 Claim ID 不会被绘制。`,
  graphScopeTitle: "图谱关系如何校验？",
  graphScopeBody: "“论文 → 证据”来自 evidence_db.json，“证据 → Research Gap”来自 research_gaps.json，“Research Gap／证据 → 想法”来自 raw_ideas.json。未知 ID 和非法关系类型会被拒绝，不会由界面猜测。",
  relationshipSupports: "支撑证据",
  relationshipSupportsGap: "支撑研究空白",
  relationshipPartialCoverage: "部分覆盖研究空白",
  relationshipChallengesGap: "质疑研究空白",
  relationshipEvidenceFor: "支撑想法",
  relationshipTargets: "催生想法",
  relationshipBasis: "记录的论证依据",
  configIntro: "此页面保存后续运行的默认配置。环境变量会覆盖已保存默认值；新建运行时填写的值又会覆盖前两者。",
  configured: "已配置",
  notConfigured: "未配置",
  saveCurrentGroup: "保存当前分组",
  reloaded: "已从磁盘重新加载设置。",
  loadingArtifact: "正在加载产物…",
  loadingLog: "正在加载运行日志…",
  logUnavailable: "该运行尚未记录进程输出。",
  confirmStop: "停止这个本地研究进程吗？已生成的产物会保留。",
  unknown: "未知",
  currentActivity: "当前活动",
  stageProgress: "阶段内进度",
  lastActivity: "最近活动",
  activityNow: "刚刚",
  activitySecondsAgo: (count) => `${count} 秒前`,
  activityMinutesAgo: (count) => `${count} 分钟前`,
  activityHoursAgo: (count) => `${count} 小时前`,
  activityDaysAgo: (count) => `${count} 天前`,
  phase_defining_requirements: "正在明确研究需求",
  phase_formalizing_problem: "正在形式化研究问题",
  phase_surveying_literature: "正在检索与筛选文献",
  phase_reading_papers: "正在精读入选论文",
  phase_positioning_papers: "正在分析论文定位",
  phase_expanding_literature: "正在扩展相关文献",
  phase_binding_evidence: "正在绑定论据与来源",
  phase_synthesizing_gaps: "正在综合研究空白",
  phase_mapping_design_space: "正在构建设计空间",
  phase_generating_ideas: "正在生成候选想法",
  phase_ranking_ideas: "正在排序候选想法",
  phase_debating_ideas: "正在进行对抗评审",
  phase_assessing_feasibility: "正在评估可行性",
  phase_writing_report: "正在撰写最终报告",
  phase_preparing_batches: "正在准备任务批次",
  phase_processing_batch: "正在处理任务批次",
  phase_checking_batches: "正在检查批次结果",
  phase_merging_batches: "正在合并批次结果",
  phase_searching_sources: "正在检索文献来源",
  phase_retrieving_full_text: "正在获取论文全文",
  phase_validating_stage: "正在验证本阶段产物",
  phase_recording_reflection: "正在记录阶段反思",
  phase_writing_artifact: "正在写入阶段产物",
  phase_running_subagent: "研究子智能体正在工作",
  phase_integrating_subagent: "正在整合子智能体结果",
  phase_reasoning: "正在分析研究证据",
  phase_runtime_error: "运行操作发生错误",
  unit_papers_collected: "篇论文已收集",
  unit_papers_processed: "篇论文已处理",
  unit_papers_positioned: "篇论文已完成定位分析",
  unit_batches: "个批次",
  unit_ideas_generated: "个想法已生成",
  unit_ideas_ranked: "个想法已排名",
  unit_ideas_reviewed: "个想法已评审",
  unit_ideas_assessed: "个想法已评估",
  unit_stage: "阶段已完成",
  metric_full_text: "全文成功",
  metric_failed: "失败",
  metric_batches: "批次",
  metric_claims: "论据",
  metric_gaps: "研究空白",
  metric_evidence_links: "证据连接",
  metric_axes: "设计轴",
  metric_combinations: "候选组合",
  metric_comparisons: "两两比较",
  metric_debate_rounds: "辩论轮次",
});

const VIEW_META = {
  studio: ["overviewEyebrow", "overviewTitle", "overviewDescription"],
  live: ["liveEyebrow", "liveTitle", "liveDescription"],
  results: ["reportEyebrow", "reportTitle", "reportDescription"],
  map: ["mapEyebrow", "mapTitle", "mapDescription"],
  papers: ["papersEyebrow", "papersTitle", "papersDescription"],
  evidence: ["evidenceEyebrow", "evidenceTitle", "evidenceDescription"],
  ideas: ["ideasEyebrow", "ideasTitle", "ideasDescription"],
  pipeline: ["pipelineEyebrow", "pipelineTitle", "pipelineDescription"],
  artifacts: ["artifactsEyebrow", "artifactsTitle", "artifactsDescription"],
  settings: ["settingsEyebrow", "settingsTitle", "settingsDescription"],
};

boot();

async function boot() {
  bindStaticEvents();
  applyLanguage();
  await refreshAll();
  state.pollTimer = window.setInterval(pollRuns, 2200);
  state.activityClockTimer = window.setInterval(updateRelativeTimes, 1000);
}

function bindStaticEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigate(button.dataset.view, true);
    });
  });
  document.querySelectorAll("[data-view-target]").forEach((button) => {
    button.addEventListener("click", () => navigate(button.dataset.viewTarget, true));
  });
  refs.navToggle.addEventListener("click", () => setNavOpen(!state.navOpen));
  refs.navScrim.addEventListener("click", () => setNavOpen(false));
  refs.newRunButton.addEventListener("click", () => {
    state.studioDraft.mode = "new";
    navigate("studio", true);
    window.setTimeout(() => document.querySelector("#researchPrompt")?.focus(), 0);
  });
  refs.refreshButton.addEventListener("click", refreshAll);
  refs.languageToggle.addEventListener("click", () => {
    state.language = state.language === "zh" ? "en" : "zh";
    safeStorageSet("autoidea-language", state.language);
    applyLanguage();
    renderGlobalStatus();
    render();
  });
  refs.runSelector.addEventListener("change", async () => {
    state.selectedRunId = refs.runSelector.value;
    state.selectedGraphNode = null;
    state.focusNode = null;
    updateUrl();
    await loadSnapshot();
    render();
  });
  refs.searchInput.addEventListener("input", () => {
    state.query = refs.searchInput.value.trim().toLocaleLowerCase();
    updateUrl();
    updateNavigation();
    renderViewOnly();
  });
  refs.confidenceFilter.addEventListener("change", () => {
    state.confidence = refs.confidenceFilter.value;
    updateUrl();
    updateNavigation();
    renderViewOnly();
  });
  refs.nodeFilter.addEventListener("change", () => {
    state.nodeKind = refs.nodeFilter.value;
    updateUrl();
    updateNavigation();
    renderViewOnly();
  });
  refs.closeArtifactButton.addEventListener("click", () => refs.artifactDialog.close());
  refs.artifactDialog.addEventListener("close", stopLogPolling);
  window.addEventListener("popstate", async () => {
    state.activeView = initialParam("view") || "studio";
    state.selectedRunId = initialParam("run");
    state.query = initialParam("q").trim().toLocaleLowerCase();
    state.confidence = initialParam("confidence");
    state.nodeKind = initialParam("node");
    state.activeConfigGroup = initialParam("group") || "quick";
    await loadSnapshot();
    render();
  });
  window.addEventListener("beforeunload", (event) => {
    if (!state.studioDraft.prompt.trim() && state.configDirty.size === 0) return;
    event.preventDefault();
    event.returnValue = "";
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.navOpen) setNavOpen(false);
  });
}

async function refreshAll() {
  setStatusKey("loading");
  refs.refreshButton.disabled = true;
  try {
    await loadRuns({ chooseDefault: true });
    await Promise.all([loadSnapshot(), loadConfig()]);
    setStatusKey("loaded");
    render();
  } catch (error) {
    handleError(error);
  } finally {
    refs.refreshButton.disabled = false;
  }
}

async function pollRuns() {
  try {
    const previousStatus = selectedRun()?.status || "";
    const runsChanged = await loadRuns({ chooseDefault: false });
    const selected = selectedRun();
    let snapshotChanged = false;
    if (selected && (ACTIVE_STATUSES.has(selected.status) || selected.status !== previousStatus)) {
      snapshotChanged = await loadSnapshot({ quiet: true });
    }
    if (!runsChanged && !snapshotChanged) return;
    updateRunContext();
    updateNavigation();
    if (!isEditingView()) {
      updatePageHeader();
      renderViewOnly({ animate: false });
    }
  } catch {
    // A transient poll failure must not replace or erase the user's form input.
  }
}

async function loadRuns({ chooseDefault = false } = {}) {
  const response = await fetch("/api/runs", { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`Runs API returned HTTP ${response.status}.`);
  const nextRuns = await response.json();
  const nextSignature = JSON.stringify(nextRuns);
  const changed = nextSignature !== state.runsSignature;
  state.runs = nextRuns;
  state.runsSignature = nextSignature;
  if (state.selectedRunId && !state.runs.some((run) => run.run_id === state.selectedRunId)) state.selectedRunId = "";
  if (chooseDefault && !state.selectedRunId && state.runs.length) {
    state.selectedRunId = state.runs.find((run) => ACTIVE_STATUSES.has(run.status))?.run_id || state.runs[0].run_id;
    updateUrl();
  }
  updateRunSelector();
  return changed;
}

async function loadSnapshot({ quiet = false } = {}) {
  if (!quiet) refs.viewRoot.setAttribute("aria-busy", "true");
  try {
    const endpoint = state.selectedRunId
      ? `/api/runs/${encodeURIComponent(state.selectedRunId)}/snapshot`
      : "/api/snapshot";
    const response = await fetch(endpoint, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Snapshot API returned HTTP ${response.status}.`);
    const nextSnapshot = await response.json();
    const nextSignature = JSON.stringify(nextSnapshot);
    const changed = nextSignature !== state.snapshotSignature;
    state.snapshot = nextSnapshot;
    state.snapshotSignature = nextSignature;
    if (!state.selectedRunId) state.rootSnapshot = state.snapshot;
    refs.workspaceName.textContent = state.snapshot.workspace.path;
    refs.workspaceName.title = state.snapshot.workspace.path;
    return changed;
  } finally {
    refs.viewRoot.setAttribute("aria-busy", "false");
  }
}

async function loadConfig() {
  const response = await fetch("/api/config", { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`Config API returned HTTP ${response.status}.`);
  const previousConfig = state.config;
  state.config = await response.json();
  if (!Object.keys(state.configDraft).length) {
    Object.values(state.config.fields || {}).forEach((field) => {
      state.configDraft[field.key] = field.value;
    });
  }
  syncStudioDefaults(previousConfig);
}

function syncStudioDefaults(previousConfig = null, changedKeys = null) {
  if (!state.config?.fields) return;
  const fields = state.config.fields;
  const previousFields = previousConfig?.fields || {};
  const bindings = [
    ["model", "model", (value) => value || ""],
    ["provider", "provider", (value) => value || ""],
    ["seed_papers_file", "seedPapers", (value) => value || ""],
    ["seed_ideas_file", "seedIdeas", (value) => value || ""],
    ["auto_approve", "autoApprove", (value) => Boolean(value)],
    ["show_thinking", "showThinking", (value) => value !== false],
  ];
  bindings.forEach(([configKey, draftKey, normalize]) => {
    if (changedKeys && !changedKeys.has(configKey)) return;
    const nextValue = normalize(fields[configKey]?.effective_value);
    const previousValue = normalize(previousFields[configKey]?.effective_value);
    const stillUsingPreviousDefault = !state.studioDraftInitialized
      || !previousConfig
      || state.studioDraft[draftKey] === previousValue;
    if (stillUsingPreviousDefault) state.studioDraft[draftKey] = nextValue;
  });
  state.studioDraftInitialized = true;
}

function render() {
  if (!state.snapshot) return;
  cleanupGraph();
  applyLanguage();
  updateRunSelector();
  updateRunContext();
  updateNavigation();
  updatePageHeader();
  updateContextTools();
  renderViewOnly();
}

function renderViewOnly({ animate = true } = {}) {
  if (!state.snapshot) return;
  const previousLog = refs.viewRoot.querySelector(".log-box");
  const logPosition = previousLog ? {
    top: previousLog.scrollTop,
    nearBottom: previousLog.scrollHeight - previousLog.scrollTop - previousLog.clientHeight < 48,
  } : null;
  cleanupGraph();
  const renderers = {
    studio: renderOverview,
    live: renderLive,
    results: renderResults,
    map: renderMap,
    papers: renderPapers,
    evidence: renderEvidence,
    ideas: renderIdeas,
    pipeline: renderPipeline,
    artifacts: renderArtifacts,
    settings: renderSettings,
  };
  const viewClass = animate ? "view-enter" : "view-content";
  refs.viewRoot.innerHTML = `<div class="${viewClass}">${(renderers[state.activeView] || renderOverview)()}</div>`;
  refs.viewRoot.setAttribute("aria-busy", "false");
  bindViewEvents();
  updateRelativeTimes();
  if (state.activeView === "map") mountGraph();
  if (state.activeView === "results") mountInlineReport();
  if (logPosition) {
    const currentLog = refs.viewRoot.querySelector(".log-box");
    if (currentLog) currentLog.scrollTop = logPosition.nearBottom ? currentLog.scrollHeight : logPosition.top;
  }
}

function updatePageHeader() {
  refs.pageHeader.hidden = state.activeView === "studio";
  if (refs.pageHeader.hidden) return;
  const meta = VIEW_META[state.activeView] || VIEW_META.studio;
  refs.pageEyebrow.textContent = t(meta[0]);
  refs.pageTitle.textContent = t(meta[1]);
  refs.pageDescription.textContent = t(meta[2]);
  refs.pageActions.innerHTML = pageActionsForView();
}

function pageActionsForView() {
  if (state.activeView === "live" && selectedRun()) {
    const run = selectedRun();
    return `${ACTIVE_STATUSES.has(run.status) ? buttonHtml("stop-selected", t("stopRun"), "button-danger") : ""}${buttonHtml("open-selected-log", t("openFullLog"), "button-secondary")}`;
  }
  if (state.activeView === "results" && artifactByPath("final_report.md")) return buttonHtml("open-final-report", t("openReport"), "button-primary");
  if (["pipeline", "artifacts"].includes(state.activeView)) return buttonHtml("refresh-view", t("refresh"), "button-secondary");
  return "";
}

function updateContextTools() {
  refs.contextTools.hidden = !FILTER_VIEWS.has(state.activeView);
  refs.confidenceControl.hidden = !["evidence", "map"].includes(state.activeView);
  refs.nodeControl.hidden = state.activeView !== "map";
  refs.searchInput.placeholder = t("searchPlaceholder");
  refs.searchInput.value = state.query;
  refs.confidenceFilter.value = state.confidence;
  refs.nodeFilter.value = state.nodeKind;
}

function updateNavigation() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    const active = button.dataset.view === state.activeView;
    button.classList.toggle("active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
    const destination = new URL(window.location.href);
    if (button.dataset.view === "studio") destination.searchParams.delete("view");
    else destination.searchParams.set("view", button.dataset.view);
    button.href = `${destination.pathname}${destination.search}${destination.hash}`;
  });
  const pending = state.runs.filter((run) => run.status === "waiting_for_input").length;
  refs.liveBadge.hidden = pending === 0;
  refs.liveBadge.textContent = formatNumber(pending);
}

function updateRunSelector() {
  const current = state.selectedRunId;
  const options = [`<option value="">${escapeHtml(t("rootWorkspace"))}</option>`];
  state.runs.forEach((run) => {
    const label = `${run.run_name || run.run_id} · ${statusLabel(run.status)}`;
    options.push(`<option value="${escapeAttr(run.run_id)}">${escapeHtml(label)}</option>`);
  });
  const markup = options.join("");
  if (state.runSelectorMarkup !== markup) {
    state.runSelectorMarkup = markup;
    refs.runSelector.innerHTML = markup;
  }
  if (refs.runSelector.value !== current) refs.runSelector.value = current;
}

function updateRunContext() {
  const run = selectedRun();
  refs.runStatus.className = `status-pill ${escapeAttr(run?.status || "neutral")}`;
  refs.runStatus.textContent = run ? statusLabel(run.status) : t("noRun");
}

function renderOverview() {
  const run = selectedRun();
  const pipeline = state.snapshot.pipeline || {};
  const counts = state.snapshot.counts || {};
  const traceState = run?.status === "pipeline_completed" ? t("traceComplete")
    : run?.status === "waiting_for_input" ? t("traceWaiting")
      : run && ACTIVE_STATUSES.has(run.status) ? t("traceRunning") : t("traceAwaiting");
  return `
    <section class="observatory-hero" aria-labelledby="heroTitle">
      <div class="hero-copy">
        <p class="eyebrow">${t("heroEyebrow")}</p>
        <h1 id="heroTitle">${t("heroTitle")}</h1>
        <p>${t("heroBody")}</p>
        <div class="hero-actions">
          <button class="button button-primary" type="button" data-action="focus-composer">${icon("plus")}${t("startResearch")}</button>
          ${run ? `<button class="button button-secondary" type="button" data-action="view-selected-run">${icon("pulse")}${t("observeRun")}</button>` : ""}
        </div>
      </div>
      <div class="trace-panel">
        <div class="trace-header"><div><p>${t("traceTitle")}</p><strong>${escapeHtml(traceState)}</strong></div><span class="status-pill ${escapeAttr(run?.status || "neutral")}">${escapeHtml(run ? statusLabel(run.status) : t("noRun"))}</span></div>
        ${renderResearchTrace(pipeline)}
        <div class="trace-legend"><span style="color:var(--primitive-teal-200)"><i></i>${t("complete")}</span><span style="color:var(--primitive-cobalt-600)"><i></i>${t("progress")}</span><span style="color:var(--primitive-amber-100)"><i></i>${t("checkpoint")}</span></div>
      </div>
    </section>
    <section class="metric-strip" aria-label="Research workspace metrics">
      ${metric(t("countPapers"), counts.papers || 0)}${metric(t("countClaims"), counts.claims || 0)}${metric(t("countIdeas"), counts.ideas || 0)}${metric(t("countArtifacts"), counts.artifacts || 0)}${metric(t("countProgress"), `${pipeline.percent || 0}%`)}
    </section>
    <section class="section-grid">
      <div class="panel span-7" id="runComposer">${renderRunComposer()}</div>
      <div class="panel span-5">${renderRecentRuns()}</div>
    </section>
    ${pipeline.persisted_state_stale ? `<aside class="panel subtle"><strong>${t("staleState")}</strong><p class="card-copy">${escapeHtml(`${pipeline.persisted_next_stage || "—"} → ${pipeline.next_stage || "—"}`)}</p></aside>` : ""}
  `;
}

function renderResearchTrace(pipeline) {
  const stages = Array.isArray(pipeline.stages) ? pipeline.stages : [];
  if (!stages.length) return `<div class="trace-canvas"><p class="trace-meta">${t("noActiveRun")}</p></div>`;
  const points = stages.map((stage, index) => {
    if (index < 7) return { x: 30 + index * 80, y: 68 };
    return { x: 510 - (index - 7) * 80, y: 216 };
  });
  const segments = points.slice(1).map((point, index) => {
    const previous = points[index];
    const complete = stages[index]?.status === "complete" && stages[index + 1]?.status === "complete";
    return `<line class="trace-route ${complete ? "complete" : ""}" x1="${previous.x}" y1="${previous.y}" x2="${point.x}" y2="${point.y}" />`;
  }).join("");
  const nodes = stages.map((stage, index) => {
    const point = points[index];
    const isCheckpoint = Boolean(stage.checkpoint);
    const shape = isCheckpoint
      ? `<rect x="${point.x - 8}" y="${point.y - 8}" width="16" height="16" rx="4" />`
      : `<circle cx="${point.x}" cy="${point.y}" r="7" />`;
    const labelY = index < 7 ? point.y - 20 : point.y + 31;
    return `<g class="trace-node ${escapeAttr(stage.status || "pending")}" data-action="trace-stage" data-stage-id="${escapeAttr(stage.id)}" role="button" tabindex="0" aria-label="${escapeAttr(`${stage.number} ${stage.name}: ${stage.status}`)}">${shape}<text class="trace-number" x="${point.x}" y="${point.y + 2.5}">${escapeHtml(stage.number)}</text><text class="trace-label" x="${point.x}" y="${labelY}" text-anchor="middle">${escapeHtml(traceStageLabel(stage.id, stage.name))}</text></g>`;
  }).join("");
  const accessibleStages = stages.map((stage) => `<li>${escapeHtml(stage.number)} ${escapeHtml(stage.name)} — ${escapeHtml(statusLabel(stage.status))}${stage.checkpoint ? ` — ${t("checkpoint")}` : ""}</li>`).join("");
  return `<div class="trace-canvas"><svg viewBox="0 0 540 280" role="img" aria-labelledby="traceSvgTitle traceSvgDesc"><title id="traceSvgTitle">${t("traceTitle")}</title><desc id="traceSvgDesc">${formatNumber(pipeline.completed_count || 0)} ${t("complete")}, ${formatNumber(pipeline.total_stages || stages.length)} ${t("stages")}</desc>${segments}${nodes}</svg><ol class="visually-hidden">${accessibleStages}</ol></div><p class="trace-meta">${formatNumber(pipeline.completed_count || 0)} / ${formatNumber(pipeline.total_stages || stages.length)} ${t("stages")} · ${formatNumber((pipeline.completion?.checkpoint_events || []).length)} / 3 ${t("checkpoints")}</p>`;
}

function renderRunComposer() {
  const d = state.studioDraft;
  const selected = selectedRun();
  const selectedHelp = d.mode === "new" ? "" : `<p class="helper">${t("selectedContext")}</p>`;
  return `
    <div class="section-header"><div><p class="section-label">${t("newRun")}</p><h2>${t("composeTitle")}</h2><p>${t("composeBody")}</p></div></div>
    <form id="runForm" class="run-composer">
      <fieldset class="field" style="border:0;padding:0;margin:0"><legend class="visually-hidden">${t("runMode")}</legend>
        <div class="mode-selector">
          ${modeRadio("new", t("modeNew"), d.mode === "new", false)}
          ${modeRadio("resume", t("modeResume"), d.mode === "resume", !selected)}
          ${modeRadio("followup", t("modeFollowup"), d.mode === "followup", !selected)}
        </div>${selectedHelp}
      </fieldset>
      <label class="field field-wide" for="researchPrompt"><span>${t("researchQuestion")}</span><textarea id="researchPrompt" name="prompt" required placeholder="${escapeAttr(t("promptPlaceholder"))}">${escapeHtml(d.prompt)}</textarea></label>
      <div class="form-grid">
        <label class="field" for="runName"><span>${t("runName")}</span><input id="runName" name="runName" value="${escapeAttr(d.runName)}" placeholder="${escapeAttr(t("runNamePlaceholder"))}" autocomplete="off" /></label>
        <label class="field" for="workspaceOverride"><span>${t("workspaceOverride")}</span><input id="workspaceOverride" name="workspace" value="${escapeAttr(d.workspace)}" placeholder="${escapeAttr(t("workspacePlaceholder"))}" autocomplete="off" /></label>
      </div>
      <details class="advanced-options"><summary>${t("advancedOptions")}</summary>
        <div class="form-grid">
          <label class="field" for="runProvider"><span>${t("provider")}</span><input id="runProvider" name="provider" value="${escapeAttr(d.provider)}" placeholder="${escapeAttr(t("configuredDefault"))}" autocomplete="off" /></label>
          <label class="field" for="runModel"><span>${t("model")}</span><input id="runModel" name="model" value="${escapeAttr(d.model)}" placeholder="${escapeAttr(t("configuredDefault"))}" autocomplete="off" /></label>
          <label class="field" for="seedPapers"><span>${t("seedPapers")}</span><input id="seedPapers" name="seedPapers" value="${escapeAttr(d.seedPapers)}" placeholder="/path/to/papers.json" autocomplete="off" /></label>
          <label class="field" for="seedIdeas"><span>${t("seedIdeas")}</span><input id="seedIdeas" name="seedIdeas" value="${escapeAttr(d.seedIdeas)}" placeholder="/path/to/ideas.md" autocomplete="off" /></label>
          <p class="helper field-wide run-overrides-help">${t("runOverridesHelp")}</p>
        </div>
        <div class="switch-row">
          <label class="checkbox-field"><input name="autoApprove" type="checkbox" ${d.autoApprove ? "checked" : ""} /><span>${t("autoApprove")}</span></label>
          <label class="checkbox-field"><input name="showThinking" type="checkbox" ${d.showThinking ? "checked" : ""} /><span>${t("showThinking")}</span></label>
        </div><p class="helper auto-approve-help">${t("autoApproveHelp")}</p>
      </details>
      <div class="section-actions"><button class="button button-primary" type="submit">${icon("arrow")}${d.mode === "resume" ? t("resumeRun") : d.mode === "followup" ? t("followupRun") : t("launchRun")}</button></div>
    </form>
  `;
}

function renderRecentRuns() {
  const runs = state.runs.slice(0, 6);
  return `<div class="section-header"><div><p class="section-label">${t("workspace")}</p><h2>${t("recentRuns")}</h2><p>${t("recentRunsBody")}</p></div></div>${runs.length ? `<div class="run-list">${runs.map((run) => renderRunCard(run, true)).join("")}</div>` : emptyState("00", t("noRuns"), t("createFirst"))}`;
}

function renderRunCard(run, compact = false) {
  const selected = run.run_id === state.selectedRunId;
  const percent = Math.round(((run.completed_stages || 0) / (run.total_stages || 14)) * 100);
  return `<article class="run-card ${selected ? "selected" : ""} content-auto" data-run-card="${escapeAttr(run.run_id)}">
    <div class="run-card-header"><div><p class="run-id" translate="no">${escapeHtml(run.run_id)}</p><h3>${escapeHtml(run.run_name || shortLabel(run.prompt, 42) || run.run_id)}</h3></div><span class="status-pill ${escapeAttr(run.status)}">${escapeHtml(statusLabel(run.status))}</span></div>
    ${compact ? "" : `<p class="run-prompt">${escapeHtml(run.prompt)}</p>`}
    <div class="run-progress" role="progressbar" aria-label="${escapeAttr(t("progress"))}" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100"><span style="--progress:${percent}%"></span></div>
    <div class="meta-row" style="margin-top:12px"><span class="meta">${formatNumber(run.completed_stages || 0)}/${formatNumber(run.total_stages || 14)} ${t("stages")} · ${escapeHtml(stageLabel(run.current_stage))}</span><button class="button button-quiet compact" type="button" data-action="select-run" data-run-id="${escapeAttr(run.run_id)}">${t("viewRun")}</button></div>
  </article>`;
}

function renderLive() {
  const run = selectedRun();
  if (!run) return emptyState("RUN", t("chooseRun"), t("chooseRunBody"), `<button class="button button-primary" data-action="new-run" type="button">${t("newRun")}</button>`);
  const completion = run.completion || {};
  const progress = run.progress || {};
  const percent = Math.round(((run.completed_stages || 0) / (run.total_stages || 14)) * 100);
  return `
    ${run.interaction ? renderInteraction(run) : ""}
    <section class="section-grid">
      <article class="panel span-8">
        <div class="section-header"><div><p class="section-label">${t("runDetail")}</p><h2>${escapeHtml(run.run_name || run.run_id)}</h2><p>${escapeHtml(run.status_detail || run.prompt)}</p></div><span class="status-pill ${escapeAttr(run.status)}">${escapeHtml(statusLabel(run.status))}</span></div>
        <div class="completion-proof">
          ${proofItem(t("progress"), `${formatNumber(run.completed_stages || 0)} / ${formatNumber(run.total_stages || 14)}`, percent === 100 ? "pass" : "pending")}
          ${proofItem(t("stage"), stageLabel(run.current_stage), run.status === "failed" ? "fail" : "pending")}
          ${proofItem(t("checkpointProof"), `${formatNumber((completion.checkpoint_events || []).length)} / 3`, (completion.checkpoint_events || []).length === 3 ? "pass" : "pending")}
        </div>
        ${renderStageProgress(progress)}
        <div class="run-progress" style="margin-top:24px" role="progressbar" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100"><span style="--progress:${percent}%"></span></div>
        <dl class="run-facts">${fact(t("thread"), run.thread_id || "—")}${fact(t("process"), run.pid || "—")}${fact(t("started"), formatDate(run.started_at))}${fact(t("finished"), formatDate(run.finished_at))}</dl>
      </article>
      <aside class="panel subtle span-4">${renderCompletionProof(completion)}</aside>
    </section>
    <section class="panel">
      <div class="section-header"><div><p class="section-label">${t("process")}</p><h2>${t("log")}</h2><p>${t("logBody")}</p></div><button class="button button-secondary compact" type="button" data-action="open-log" data-run-id="${escapeAttr(run.run_id)}">${t("openFullLog")}</button></div>
      <pre class="log-box" tabindex="0">${escapeHtml(tailText(run.log_tail || "", 12000) || "Waiting for process output…")}</pre>
    </section>
  `;
}

function renderInteraction(run) {
  const interaction = run.interaction || {};
  if (interaction.kind === "tool_approval") {
    const actions = Array.isArray(interaction.actions) ? interaction.actions : [];
    return `<section class="checkpoint-panel" aria-labelledby="interactionTitle"><div class="checkpoint-header"><div class="inline-cluster"><span class="checkpoint-mark">HITL</span><div><p class="section-label">${t("toolApproval")}</p><h2 id="interactionTitle">${t("inputRequired")}</h2></div></div><span class="status-pill waiting_for_input">${statusLabel("waiting_for_input")}</span></div><p class="card-copy">${t("toolApprovalBody")}</p><div class="checkpoint-questions">${actions.map((action) => `<article class="research-card"><h3 translate="no">${escapeHtml(action.name || "action")}</h3><pre class="text-viewer">${escapeHtml(JSON.stringify(action.args || {}, null, 2))}</pre></article>`).join("")}</div><div class="interaction-actions"><button class="button button-secondary" type="button" data-action="tool-decision" data-decision="reject" data-run-id="${escapeAttr(run.run_id)}">${t("reject")}</button><button class="button button-primary" type="button" data-action="tool-decision" data-decision="approve" data-run-id="${escapeAttr(run.run_id)}">${t("approve")}</button></div></section>`;
  }
  if (["checkpoint", "ask_user"].includes(interaction.kind)) {
    const questions = Array.isArray(interaction.questions) ? interaction.questions : [];
    return `<section class="checkpoint-panel" aria-labelledby="interactionTitle"><div class="checkpoint-header"><div class="inline-cluster"><span class="checkpoint-mark">${escapeHtml(interaction.checkpoint_stage ? interaction.checkpoint_stage.replace("stage_", "S") : "ASK")}</span><div><p class="section-label">${interaction.checkpoint_stage ? t("checkpoint") : t("inputRequired")}</p><h2 id="interactionTitle">${t("inputRequired")}</h2></div></div><span class="status-pill waiting_for_input">${statusLabel("waiting_for_input")}</span></div><p class="card-copy">${t("inputBody")}</p><form id="interactionForm" data-run-id="${escapeAttr(run.run_id)}"><div class="checkpoint-questions">${questions.map(renderQuestion).join("")}</div><div class="interaction-actions"><button class="button button-primary" type="submit">${t("submitDecision")}</button></div></form></section>`;
  }
  if (interaction.kind === "checkpoint_review") return renderLegacyCheckpoint(run, interaction);
  if (["multiple_choice", "text"].includes(interaction.kind)) return renderLegacyInput(run, interaction);
  return "";
}

function renderQuestion(question, index) {
  const required = question.required !== false;
  const runId = selectedRun()?.run_id || "run";
  const savedAnswer = state.interactionDraft[`${runId}:answer-${index}`] || "";
  const savedOther = state.interactionDraft[`${runId}:other-${index}`] || "";
  const prompt = `<p id="question-${index}">${escapeHtml(question.question || `${t("inputRequired")} ${index + 1}`)}${required ? "" : ` <span class="meta">(${t("optional")})</span>`}</p>`;
  if (question.type === "multiple_choice") {
    const choices = Array.isArray(question.choices) ? question.choices : [];
    return `<fieldset class="question-block"><legend class="visually-hidden">${escapeHtml(question.question || "Question")}</legend>${prompt}<div class="choice-list">${choices.map((choice, choiceIndex) => {
      const value = choice.value ?? choice.label ?? String(choice);
      const label = choice.label ?? value;
      return `<label class="choice-option"><input type="radio" name="answer-${index}" value="${escapeAttr(value)}" ${savedAnswer === String(value) ? "checked" : ""} ${required && choiceIndex === 0 ? "required" : ""} /><span>${escapeHtml(label)}</span></label>`;
    }).join("")}<label class="choice-option"><input type="radio" name="answer-${index}" value="__other__" ${savedAnswer === "__other__" ? "checked" : ""} ${required && choices.length === 0 ? "required" : ""} /><span>${t("otherAnswer")}</span></label></div><label class="field"><span class="visually-hidden">${t("otherAnswer")}</span><input name="other-${index}" value="${escapeAttr(savedOther)}" placeholder="${escapeAttr(t("otherAnswer"))}…" autocomplete="off" /></label></fieldset>`;
  }
  return `<label class="question-block" for="answer-${index}">${prompt}<textarea id="answer-${index}" name="answer-${index}" ${required ? "required" : ""}>${escapeHtml(savedAnswer)}</textarea></label>`;
}

function renderLegacyInput(run, interaction) {
  const options = Array.isArray(interaction.options) ? interaction.options : [];
  const saved = state.interactionDraft[`${run.run_id}:legacy-answer`] || "";
  return `<section class="checkpoint-panel"><div class="checkpoint-header"><div><p class="section-label">${t("inputRequired")}</p><h2>${escapeHtml(interaction.question || interaction.prompt || t("inputRequired"))}</h2></div></div><form id="legacyInputForm" data-run-id="${escapeAttr(run.run_id)}"><div class="checkpoint-questions">${options.length ? `<div class="choice-list">${options.map((option) => `<label class="choice-option"><input type="radio" name="legacy-answer" value="${escapeAttr(option.key)}" ${saved === option.key ? "checked" : ""} required /><span>${escapeHtml(option.label)}</span></label>`).join("")}</div>` : `<label class="field"><span>${t("otherAnswer")}</span><textarea name="legacy-answer" required>${escapeHtml(saved)}</textarea></label>`}</div><div class="interaction-actions"><button class="button button-primary" type="submit">${t("submitDecision")}</button></div></form></section>`;
}

function renderLegacyCheckpoint(run, interaction) {
  const options = Array.isArray(interaction.options) ? interaction.options : [];
  const saved = state.interactionDraft[`${run.run_id}:feedback`] || "";
  return `<section class="checkpoint-panel"><div class="checkpoint-header"><div><p class="section-label">${t("checkpoint")}</p><h2>${escapeHtml(interaction.question || t("inputRequired"))}</h2></div></div><form id="legacyCheckpointForm" data-run-id="${escapeAttr(run.run_id)}"><label class="field"><span>${t("otherAnswer")} (${t("optional")})</span><textarea name="feedback">${escapeHtml(saved)}</textarea></label><div class="interaction-actions">${options.map((option) => `<button class="button ${option.key === "approve" ? "button-primary" : "button-secondary"}" type="submit" name="action" value="${escapeAttr(option.key)}">${escapeHtml(option.label)}</button>`).join("")}</div></form></section>`;
}

function renderResults() {
  const run = selectedRun();
  if (!run) return emptyState("12", t("chooseRun"), t("chooseRunBody"), `<button class="button button-primary" data-action="new-run" type="button">${t("newRun")}</button>`);
  const pipeline = state.snapshot.pipeline || {};
  const completion = pipeline.completion || run.completion || {};
  const report = artifactByPath("final_report.md");
  const ideas = Array.isArray(state.snapshot.ideas) ? state.snapshot.ideas.slice(0, 3) : [];
  const support = (state.snapshot.artifacts || []).filter((artifact) => artifact.path !== "final_report.md").slice(-6);
  return `
    <section class="section-grid">
      <article class="panel span-8 report-panel">
        <div class="section-header"><div><p class="section-label">${t("finalReport")}</p><h2>${report ? t("reportReady") : t("finalNotReady")}</h2><p>${report ? t("reportReading") : t("finalNotReadyBody")}</p></div>${report ? `<span class="status-pill ${completion.verified ? "pipeline_completed" : "checkpoint_reached"}">${completion.verified ? t("verified") : t("notVerified")}</span>` : ""}</div>
        ${report ? `<article id="inlineReport" class="markdown-body report-reading-surface" aria-busy="true"><p class="meta">${t("loadingReport")}</p></article>` : emptyState("12", t("finalNotReady"), t("finalNotReadyBody"), `<button class="button button-secondary" type="button" data-action="view-selected-run">${t("observeRun")}</button>`)}
      </article>
      <aside class="panel subtle span-4">${renderCompletionProof(completion)}</aside>
    </section>
    <section class="section-grid">
      <article class="panel span-7">
        <div class="section-header"><div><p class="section-label">${t("topIdeas")}</p><h2>${t("topIdeas")}</h2></div></div>
        ${ideas.length ? `<div class="run-list">${ideas.map((idea, index) => renderIdeaCard(idea, index, true)).join("")}</div>` : `<p class="card-copy">${t("noIdeas")}</p>`}
      </article>
      <aside class="panel span-5">
        <div class="section-header"><div><p class="section-label">${t("supportingArtifacts")}</p><h2>${t("supportingArtifacts")}</h2></div></div>
        ${support.length ? `<div class="run-list">${support.map((artifact) => artifactRow(artifact)).join("")}</div>` : `<p class="card-copy">${t("noArtifacts")}</p>`}
      </aside>
    </section>`;
}

function renderMap() {
  const graph = buildEnhancedGraph();
  if (!graph.nodes.length) return emptyState("MAP", t("noMatches"), t("mapHint"));
  const provenance = graphProvenanceStats();
  const selected = graph.nodes.find((node) => node.id === state.selectedGraphNode) || null;
  const connections = selected
    ? graph.edges.filter((edge) => edge.source === selected.id || edge.target === selected.id)
    : [];
  const nodeLookup = new Map(graph.nodes.map((node) => [node.id, node]));
  const relationshipRows = graph.edges.map((edge) => {
    const source = nodeLookup.get(edge.source);
    const target = nodeLookup.get(edge.target);
    return `<tr><td>${escapeHtml(source?.label || edge.source)}</td><td>${escapeHtml(edgeKindLabel(edge.kind))}</td><td>${escapeHtml(target?.label || edge.target)}</td><td>${escapeHtml(edge.detail || "—")}</td></tr>`;
  }).join("");
  return `
    <div class="provenance-block">
      <section class="provenance-status ${provenance.unmappedClaims ? "warning" : "complete"}" aria-label="${escapeAttr(t("paperEvidenceMapping"))}">
        <div class="provenance-score"><span>${t("paperEvidenceMapping")}</span><strong>${formatNumber(provenance.mappedClaims)}/${formatNumber(provenance.totalClaims)}</strong></div>
        <p>${provenance.unmappedClaims ? t("provenanceMissing", provenance.unmappedClaims) : t("provenanceComplete")} ${t("provenancePaperUse", provenance.usedPapers, provenance.totalPapers, provenance.unusedPapers)}</p>
      </section>
      <section class="provenance-status ${provenance.unmappedGaps ? "warning" : "complete"}" aria-label="${escapeAttr(t("evidenceGapMapping"))}">
        <div class="provenance-score"><span>${t("evidenceGapMapping")}</span><strong>${formatNumber(provenance.mappedGaps)}/${formatNumber(provenance.totalGaps)}</strong></div>
        <p>${provenance.unmappedGaps ? t("gapProvenanceMissing", provenance.unmappedGaps) : t("gapProvenanceComplete")}</p>
      </section>
      <details class="provenance-scope"><summary>${t("graphScopeTitle")}</summary><p>${t("graphScopeBody")}</p></details>
    </div>
    <section class="map-layout">
      <div class="graph-stage">
        <div class="graph-toolbar" aria-label="${escapeAttr(t("actions"))}">
          <button class="icon-control" type="button" data-action="graph-zoom-in" title="${escapeAttr(t("zoomIn"))}" aria-label="${escapeAttr(t("zoomIn"))}">${icon("plus")}</button>
          <button class="icon-control" type="button" data-action="graph-zoom-out" title="${escapeAttr(t("zoomOut"))}" aria-label="${escapeAttr(t("zoomOut"))}">${icon("minus")}</button>
          <button class="icon-control" type="button" data-action="graph-reset" title="${escapeAttr(t("resetView"))}" aria-label="${escapeAttr(t("resetView"))}">${icon("reset")}</button>
        </div>
        <svg id="researchGraph" class="graph-svg" viewBox="0 0 900 640" role="group" aria-label="${escapeAttr(t("mapTitle"))}">
          <defs><marker id="graphArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z"></path></marker></defs>
          <g id="graphCanvas"><g id="graphEdges"></g><g id="graphNodes"></g></g>
        </svg>
      </div>
      <aside class="map-inspector" id="mapInspector">
        <div class="legend-grid" aria-label="${escapeAttr(t("nodeType"))}">${["paper", "claim", "gap", "idea"].map((kind) => `<span class="legend-item"><i class="swatch" style="background:${graphNodeColor(kind)}"></i>${escapeHtml(nodeKindLabel(kind))}</span>`).join("")}</div>
        ${selected ? `<p class="section-label">${t("selectedNode")}</p><h2>${escapeHtml(selected.label)}</h2><p class="card-copy"><span class="tag">${escapeHtml(nodeKindLabel(selected.kind))}</span> ${escapeHtml(selected.group || "")}</p><p class="card-copy">${formatNumber(connections.length)} ${t("connections")}</p><div class="section-actions">${state.focusNode ? `<button class="button button-secondary compact" type="button" data-action="clear-graph-focus">${t("clearFocus")}</button>` : `<button class="button button-primary compact" type="button" data-action="focus-graph-node" data-node-id="${escapeAttr(selected.id)}">${t("focusNeighborhood")}</button>`}</div>` : `<p class="section-label">${t("mapInspector")}</p><h2>${t("selectGraphNode")}</h2><p class="card-copy">${t("mapHint")}</p>`}
        <p class="helper">${t("graphInstructions")}</p>
      </aside>
    </section>
    <p class="meta">${t("showingNodes", graph.nodes.length, graph.edges.length)}</p>
    <details class="accessible-graph-list"><summary>${t("accessibleRelationships")} · ${formatNumber(graph.edges.length)}</summary>
      <div class="data-table-wrap" tabindex="0"><table><thead><tr><th scope="col">${t("relationshipFrom")}</th><th scope="col">${t("relationshipType")}</th><th scope="col">${t("relationshipTo")}</th><th scope="col">${t("relationshipBasis")}</th></tr></thead><tbody>${relationshipRows}</tbody></table></div>
    </details>`;
}

function renderPapers() {
  const query = state.query;
  const papers = (state.snapshot.papers || []).filter((paper) => matchesQuery(paper, query));
  if (!papers.length) return emptyState("P", query ? t("noMatches") : t("noPapers"), t("papersDescription"));
  return `<section class="cards">${papers.map((paper, index) => {
    const position = paper.position || {};
    const sourceLink = safeHref(paper.url);
    return `<article class="research-card content-auto">
      <div class="meta-row"><span class="card-index">${escapeHtml(paper.paper_id || `P${index + 1}`)}</span>${paper.year ? `<span class="meta">${formatNumber(paper.year)}</span>` : ""}</div>
      <h2>${escapeHtml(paper.title)}</h2>
      <p class="card-copy">${escapeHtml(paper.relevance || position.summary || "")}</p>
      <dl class="run-facts">${fact(t("authors"), (paper.authors || []).join(", ") || "—")}${fact(t("venue"), paper.venue || paper.source || "—")}${fact(t("weakestLink"), position.weakest_link || "—")}</dl>
      ${position.initial_attack ? `<details class="advanced-options"><summary>${t("initialAttack")}</summary><p class="card-copy">${escapeHtml(position.initial_attack)}</p></details>` : ""}
      ${sourceLink ? `<div class="section-actions"><a class="button button-secondary compact" href="${escapeAttr(sourceLink)}" target="_blank" rel="noopener noreferrer">${t("source")}${icon("external")}</a></div>` : ""}
    </article>`;
  }).join("")}</section>`;
}

function renderEvidence() {
  const claims = (state.snapshot.claims || []).filter((claim) => {
    const confidenceMatches = !state.confidence || String(claim.confidence).toUpperCase() === state.confidence;
    return confidenceMatches && matchesQuery(claim, state.query);
  });
  if (!claims.length) return emptyState("C", state.query || state.confidence ? t("noMatches") : t("noEvidence"), t("evidenceDescription"));
  return `<div class="data-table-wrap" tabindex="0" aria-label="${escapeAttr(t("navEvidence"))}"><table><thead><tr><th scope="col">${t("citation")}</th><th scope="col">${t("claim")}</th><th scope="col">${t("source")}</th><th scope="col">${t("confidence")}</th><th scope="col">${t("section")}</th></tr></thead><tbody>${claims.map((claim) => {
    const sourceLink = safeHref(claim.source_url);
    const source = escapeHtml(claim.source_title || claim.source_paper_id || "—");
    return `<tr><td><span class="card-index">${escapeHtml(claim.citation_id)}</span></td><td>${escapeHtml(claim.claim)}</td><td>${sourceLink ? `<a href="${escapeAttr(sourceLink)}" target="_blank" rel="noopener noreferrer">${source}</a>` : source}</td><td><span class="tag ${escapeAttr(String(claim.confidence || "").toLowerCase())}">${escapeHtml(claim.confidence || "—")}</span></td><td>${escapeHtml(claim.section || claim.evidence_type || "—")}</td></tr>`;
  }).join("")}</tbody></table></div>`;
}

function renderIdeas() {
  const ideas = (state.snapshot.ideas || []).filter((idea) => matchesQuery(idea, state.query));
  const axes = Array.isArray(state.snapshot.design_axes) ? state.snapshot.design_axes : [];
  return `${ideas.length ? `<section class="cards">${ideas.map((idea, index) => renderIdeaCard(idea, index)).join("")}</section>` : emptyState("I", state.query ? t("noMatches") : t("noIdeas"), t("ideasDescription"))}
    ${axes.length ? `<section class="panel"><div class="section-header"><div><p class="section-label">${t("designSpace")}</p><h2>${t("designSpace")}</h2></div></div><div class="cards">${axes.map((axis) => `<article class="research-card"><h3>${escapeHtml(axis.name)}</h3><p class="card-copy">${escapeHtml(axis.description)}</p><div class="tag-row">${(axis.explored || []).map((item) => `<span class="tag complete">${escapeHtml(item)}</span>`).join("")}${(axis.unexplored || []).map((item) => `<span class="tag waiting_for_input">${escapeHtml(item)}</span>`).join("")}</div></article>`).join("")}</div></section>` : ""}`;
}

function renderIdeaCard(idea, index, compact = false) {
  const score = idea.composite_score == null ? "—" : formatNumber(idea.composite_score, 1);
  const assessment = Object.entries(idea.self_assessment || {});
  return `<article class="research-card content-auto">
    <div class="meta-row"><span class="card-index">${escapeHtml(idea.idea_id || `I${index + 1}`)}</span><span class="tag">${t("score")} ${score}</span></div>
    <h3>${escapeHtml(idea.title)}</h3>
    <p class="card-copy">${escapeHtml(idea.one_liner || (!compact ? idea.description : ""))}</p>
    ${!compact && idea.description && idea.description !== idea.one_liner ? `<p class="card-copy">${escapeHtml(idea.description)}</p>` : ""}
    ${(idea.target_gaps || []).length ? `<div class="tag-row">${idea.target_gaps.map((gap) => `<span class="tag waiting_for_input">${escapeHtml(gap)}</span>`).join("")}</div>` : ""}
    ${!compact && assessment.length ? `<dl class="run-facts">${assessment.map(([key, value]) => fact(key.replaceAll("_", " "), value)).join("")}</dl>` : ""}
    ${(idea.supporting_evidence || []).length ? `<p class="meta">${t("evidenceLinks")}: ${escapeHtml(idea.supporting_evidence.join(", "))}</p>` : ""}
  </article>`;
}

function renderPipeline() {
  const pipeline = state.snapshot.pipeline || {};
  const stages = normalizePipelineStages(pipeline);
  const completion = pipeline.completion || selectedRun()?.completion || {};
  return `
    <section class="section-grid">
      <article class="panel span-8">
        <div class="section-header"><div><p class="section-label">${t("pipelineStages")}</p><h2>${formatNumber(pipeline.completed_count || stages.filter((stage) => stage.status === "complete").length)} / ${formatNumber(pipeline.total_stages || stages.length)} ${t("stages")}</h2><p>${escapeHtml(progressActivity(pipeline.active_progress) || pipeline.active_detail || `${t("stage")}: ${stageLabel(pipeline.active_stage || pipeline.next_stage)}`)}</p></div></div>
        ${renderStageProgress(pipeline.active_progress)}
        ${stages.length ? `<div class="pipeline-list">${stages.map((stage) => renderPipelineStage(stage)).join("")}</div>` : emptyState("01", t("notStarted"), t("pipelineDescription"))}
      </article>
      <aside class="panel subtle span-4">${renderCompletionProof(completion)}${pipeline.persisted_state_stale ? `<p class="config-status warning">${t("staleState")} ${escapeHtml(`${pipeline.persisted_next_stage || "—"} → ${pipeline.next_stage || "—"}`)}</p>` : ""}</aside>
    </section>`;
}

function renderPipelineStage(stage) {
  const artifacts = stage.required_artifacts || [];
  const missing = new Set([...(stage.missing_artifacts || []), ...(stage.invalid_artifacts || [])]);
  const status = stage.status || "pending";
  const checkpointText = stage.checkpoint
    ? stage.checkpoint_recorded ? t("checkpointRecorded") : t("checkpointNotRecorded")
    : "";
  return `<article id="pipeline-${escapeAttr(stage.id)}" class="pipeline-stage ${escapeAttr(status)} ${stage.checkpoint ? "checkpoint" : ""}" tabindex="-1">
    <div class="stage-marker" aria-hidden="true">${escapeHtml(stage.number || stageNumber(stage.id))}</div>
    <div class="stage-body"><h3>${escapeHtml(localizedStageName(stage.id, stage.name))}</h3><p>${escapeHtml(statusLabel(status))}${checkpointText ? ` · ${escapeHtml(checkpointText)}` : ""}</p></div>
    <div class="stage-artifacts">${artifacts.map((path) => artifactByPath(path) ? `<button class="artifact-inline" type="button" data-action="open-artifact" data-path="${escapeAttr(path)}">${escapeHtml(path)}</button>` : `<code class="${missing.has(path) ? "missing" : ""}">${escapeHtml(path)}</code>`).join("")}</div>
  </article>`;
}

function renderArtifacts() {
  const artifacts = (state.snapshot.artifacts || []).filter((artifact) => matchesQuery(artifact, state.query));
  if (!artifacts.length) return emptyState("DIR", state.query ? t("noMatches") : t("noArtifacts"), t("artifactsDescription"));
  return `<section class="artifact-list">${artifacts.map((artifact) => `<button class="artifact-card content-auto" type="button" data-action="open-artifact" data-path="${escapeAttr(artifact.path)}"><span><span class="artifact-kind">${escapeHtml(artifact.kind)}</span><h2>${escapeHtml(artifact.title)}</h2></span><span class="artifact-meta">${escapeHtml(artifact.path)} · ${formatBytes(artifact.size_bytes)}</span></button>`).join("")}</section>`;
}

function renderSettings() {
  if (!state.config) return emptyState("CFG", t("loading"), t("settingsDescription"));
  const groups = Array.isArray(state.config.groups) ? state.config.groups : [];
  const active = groups.find((group) => group.id === state.activeConfigGroup) || groups[0];
  if (!active) return emptyState("CFG", t("notConfigured"), t("settingsDescription"));
  const fields = (active.fields || []).map((key) => state.config.fields?.[key]).filter(Boolean);
  return `<section class="config-layout">
    <nav class="config-nav" aria-label="${escapeAttr(t("navSettings"))}">${groups.map((group) => `<button type="button" class="${group.id === active.id ? "active" : ""}" data-action="config-group" data-group="${escapeAttr(group.id)}" ${group.id === active.id ? 'aria-current="page"' : ""}>${escapeHtml(state.language === "zh" ? group.title_zh || group.title : group.title)}</button>`).join("")}</nav>
    <form id="configForm" class="config-section" data-group="${escapeAttr(active.id)}">
      <div class="section-header"><div><p class="section-label">${t("configFile")}</p><h2>${escapeHtml(state.language === "zh" ? active.title_zh || active.title : active.title)}</h2><p>${t("configIntro")}</p><p class="meta" translate="no">${escapeHtml(state.config.path || "")}</p></div></div>
      <div class="config-grid">${fields.map(renderConfigField).join("")}</div>
      <p id="configStatus" class="config-status ${escapeAttr(state.configStatusKind)}" role="status">${escapeHtml(state.configStatus)}</p>
      <div class="section-actions"><button class="button button-secondary" type="button" data-action="reload-config">${t("reloadSettings")}</button><button class="button button-primary" type="submit">${t("saveCurrentGroup")}</button></div>
    </form>
  </section>`;
}

function renderConfigField(field) {
  const value = Object.hasOwn(state.configDraft, field.key) ? state.configDraft[field.key] : field.value;
  const id = `config-${field.key}`;
  const effective = field.secret
    ? (field.is_set ? `${t("currentEffective")}: ${field.masked_value || t("configured")}` : t("notConfigured"))
    : `${t("currentEffective")}: ${formatConfigValue(field.effective_value)}`;
  const override = field.env_overridden ? ` · ${t("envOverride")}: ${field.env_var}` : "";
  const effectiveClass = field.env_overridden ? "config-effective override" : "config-effective";
  if (field.type === "bool") {
    return `<div class="config-field boolean"><label class="switch-field" for="${escapeAttr(id)}"><input id="${escapeAttr(id)}" name="${escapeAttr(field.key)}" type="checkbox" data-config-key="${escapeAttr(field.key)}" ${value ? "checked" : ""} /><span>${escapeHtml(field.label)}</span></label><span class="${effectiveClass}">${escapeHtml(effective + override)}</span></div>`;
  }
  if (field.type === "select") {
    const options = Array.from(new Set([...(field.options || []), value].filter((item) => item !== "" && item != null)));
    return `<label class="config-field" for="${escapeAttr(id)}"><span class="config-label">${escapeHtml(field.label)}</span><select id="${escapeAttr(id)}" name="${escapeAttr(field.key)}" data-config-key="${escapeAttr(field.key)}">${options.map((option) => `<option value="${escapeAttr(option)}" ${String(option) === String(value) ? "selected" : ""}>${escapeHtml(option)}</option>`).join("")}</select><span class="${effectiveClass}">${escapeHtml(effective + override)}</span></label>`;
  }
  const type = field.secret ? "password" : field.type === "int" || field.type === "float" ? "number" : "text";
  const step = field.type === "float" ? "any" : field.type === "int" ? "1" : "";
  const placeholder = field.secret ? field.masked_value || t("secretHelp") : "";
  return `<label class="config-field" for="${escapeAttr(id)}"><span class="config-label">${escapeHtml(field.label)}</span><input id="${escapeAttr(id)}" name="${escapeAttr(field.key)}" type="${type}" ${step ? `step="${step}"` : ""} data-config-key="${escapeAttr(field.key)}" value="${escapeAttr(field.secret ? "" : value ?? "")}" placeholder="${escapeAttr(placeholder)}" autocomplete="${field.secret ? "new-password" : "off"}" /><span class="${effectiveClass}">${escapeHtml(effective + override)}${field.secret ? ` · ${t("secretHelp")}` : ""}</span></label>`;
}

function normalizePipelineStages(pipeline) {
  if (Array.isArray(pipeline?.stages)) return pipeline.stages;
  if (pipeline?.stages && typeof pipeline.stages === "object") {
    return Object.entries(pipeline.stages).map(([id, stage]) => ({ id, number: stageNumber(id), ...(stage || {}) }));
  }
  return [];
}

function bindViewEvents() {
  refs.viewRoot.onclick = (event) => {
    const control = event.target.closest("[data-action]");
    if (!control) return;
    handleAction(control, event).catch(handleError);
  };
  refs.viewRoot.onkeydown = (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const control = event.target.closest('[role="button"][data-action]');
    if (!control) return;
    event.preventDefault();
    handleAction(control, event).catch(handleError);
  };
  refs.pageActions.onclick = (event) => {
    const control = event.target.closest("[data-action]");
    if (!control) return;
    handleAction(control, event).catch(handleError);
  };
  refs.viewRoot.oninput = (event) => captureDynamicInput(event.target);
  refs.viewRoot.onchange = (event) => captureDynamicInput(event.target);
  refs.viewRoot.onsubmit = (event) => {
    event.preventDefault();
    handleFormSubmit(event).catch(handleError);
  };
}

async function handleAction(control, event) {
  const action = control.dataset.action;
  if (!action) return;
  if (action === "focus-composer") {
    document.querySelector("#runComposer")?.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
    window.setTimeout(() => document.querySelector("#researchPrompt")?.focus(), 80);
  } else if (action === "view-selected-run") {
    navigate("live", true);
  } else if (action === "new-run") {
    state.studioDraft.mode = "new";
    navigate("studio", true);
    window.setTimeout(() => document.querySelector("#researchPrompt")?.focus(), 0);
  } else if (action === "select-run") {
    await selectRun(control.dataset.runId, "live");
  } else if (action === "open-artifact") {
    await openArtifact(control.dataset.path);
  } else if (action === "open-log" || action === "open-selected-log") {
    await openRunLog(control.dataset.runId || state.selectedRunId);
  } else if (action === "open-final-report") {
    await openArtifact("final_report.md");
  } else if (action === "stop-selected") {
    await stopRun(state.selectedRunId);
  } else if (action === "refresh-view") {
    await loadSnapshot();
    render();
  } else if (action === "tool-decision") {
    await sendRunResponse(control.dataset.runId, { decision: control.dataset.decision || "reject" }, control);
  } else if (action === "config-group") {
    state.activeConfigGroup = control.dataset.group || "quick";
    updateUrl();
    updateNavigation();
    renderViewOnly();
  } else if (action === "reload-config") {
    await reloadConfig();
  } else if (action === "focus-graph-node") {
    state.focusNode = control.dataset.nodeId || state.selectedGraphNode;
    renderViewOnly();
  } else if (action === "clear-graph-focus") {
    state.focusNode = null;
    renderViewOnly();
  } else if (action === "graph-zoom-in") {
    zoomGraph(1.2);
  } else if (action === "graph-zoom-out") {
    zoomGraph(1 / 1.2);
  } else if (action === "graph-reset") {
    state.graph.transform = { x: 0, y: 0, k: 1 };
    applyGraphTransform();
  } else if (action === "trace-stage") {
    const stageId = control.dataset.stageId;
    navigate("pipeline", true);
    window.setTimeout(() => document.querySelector(`#pipeline-${cssEscape(stageId)}`)?.focus({ preventScroll: true }), 0);
  }
  if (event) event.stopPropagation();
}

async function handleFormSubmit(event) {
  const form = event.target;
  if (!(form instanceof HTMLFormElement)) return;
  if (form.id === "runForm") {
    await submitRun(form);
    return;
  }
  if (form.id === "interactionForm") {
    const run = selectedRun();
    const questions = Array.isArray(run?.interaction?.questions) ? run.interaction.questions : [];
    const answers = questions.map((question, index) => {
      if (question.type === "multiple_choice") {
        const selected = form.querySelector(`input[name="answer-${index}"]:checked`);
        if (!selected) return "";
        if (selected.value === "__other__") return String(form.elements.namedItem(`other-${index}`)?.value || "").trim();
        return selected.value;
      }
      return String(form.elements.namedItem(`answer-${index}`)?.value || "").trim();
    });
    const missing = questions.findIndex((question, index) => question.required !== false && !answers[index]);
    if (missing >= 0) {
      const field = form.elements.namedItem(`answer-${missing}`);
      if (field && typeof field.focus === "function") field.focus();
      throw new Error(`Question ${missing + 1} requires an answer.`);
    }
    await sendRunResponse(form.dataset.runId, { status: "answered", answers }, form.querySelector("button[type=submit]"));
    return;
  }
  if (form.id === "legacyInputForm") {
    const data = new FormData(form);
    await sendRunResponse(form.dataset.runId, { value: String(data.get("legacy-answer") || "") }, form.querySelector("button[type=submit]"));
    return;
  }
  if (form.id === "legacyCheckpointForm") {
    const data = new FormData(form);
    const action = event.submitter?.value || data.get("action") || "approve";
    await startLegacyFollowup(form.dataset.runId, String(action), String(data.get("feedback") || ""), event.submitter);
    return;
  }
  if (form.id === "configForm") await saveConfig(form);
}

function captureDynamicInput(target) {
  if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement)) return;
  const runForm = target.closest("#runForm");
  if (runForm) {
    captureStudioDraft(runForm);
    if (target.name === "mode" && target.checked) {
      const selected = selectedRun();
      if (state.studioDraft.mode !== "new" && selected) {
        state.studioDraft.model = selected.model || state.studioDraft.model;
        state.studioDraft.provider = selected.provider || state.studioDraft.provider;
        state.studioDraft.seedPapers = selected.seed_papers || state.studioDraft.seedPapers;
        state.studioDraft.seedIdeas = selected.seed_ideas || state.studioDraft.seedIdeas;
      }
      renderViewOnly();
    }
    return;
  }
  if (target.dataset.configKey) {
    const field = state.config?.fields?.[target.dataset.configKey];
    state.configDraft[target.dataset.configKey] = target.type === "checkbox" ? target.checked : target.value;
    state.configDirty.add(target.dataset.configKey);
    if (field?.secret && target.value === "") state.configDirty.delete(target.dataset.configKey);
    return;
  }
  const interactionForm = target.closest("#interactionForm, #legacyInputForm, #legacyCheckpointForm");
  if (interactionForm) {
    const key = `${interactionForm.dataset.runId || "run"}:${target.name}`;
    state.interactionDraft[key] = target.type === "radio" ? (target.checked ? target.value : state.interactionDraft[key]) : target.value;
  }
}

function captureStudioDraft(form) {
  const data = new FormData(form);
  state.studioDraft = {
    prompt: String(data.get("prompt") || ""),
    runName: String(data.get("runName") || ""),
    workspace: String(data.get("workspace") || ""),
    model: String(data.get("model") || ""),
    provider: String(data.get("provider") || ""),
    seedPapers: String(data.get("seedPapers") || ""),
    seedIdeas: String(data.get("seedIdeas") || ""),
    autoApprove: data.get("autoApprove") === "on",
    showThinking: data.get("showThinking") === "on",
    mode: String(data.get("mode") || state.studioDraft.mode || "new"),
  };
}

async function submitRun(form) {
  captureStudioDraft(form);
  const draft = state.studioDraft;
  const selected = selectedRun();
  if (draft.mode !== "new" && !selected) throw new Error(t("chooseRun"));
  const prompt = draft.prompt.trim();
  if (!prompt) throw new Error(`${t("researchQuestion")}: ${t("missing")}`);
  const inherited = draft.mode === "new" ? null : selected;
  const payload = {
    prompt,
    run_name: inherited?.run_name || draft.runName.trim(),
    workspace: inherited?.workspace || draft.workspace.trim(),
    model: draft.model.trim() || inherited?.model || "",
    provider: draft.provider.trim() || inherited?.provider || "",
    thread_id: inherited?.thread_id || "",
    seed_papers: draft.seedPapers.trim() || inherited?.seed_papers || "",
    seed_ideas: draft.seedIdeas.trim() || inherited?.seed_ideas || "",
    auto_approve: draft.autoApprove,
    show_thinking: inherited ? inherited.show_thinking !== false : draft.showThinking,
    mode: draft.mode,
    parent_run_id: inherited?.run_id || "",
  };
  const submit = form.querySelector("button[type=submit]");
  setControlBusy(submit, true);
  try {
    const run = await requestJson("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.selectedRunId = run.run_id;
    state.interactionDraft = {};
    state.studioDraft.prompt = "";
    state.studioDraft.runName = "";
    state.studioDraft.workspace = "";
    state.studioDraft.mode = "new";
    await loadRuns();
    await loadSnapshot();
    updateUrl();
    showToast(t("runStarted"));
    navigate("live", true);
  } finally {
    setControlBusy(submit, false);
  }
}

async function sendRunResponse(runId, payload, control) {
  if (!runId) throw new Error(t("chooseRun"));
  setControlBusy(control, true);
  try {
    await requestJson(`/api/runs/${encodeURIComponent(runId)}/input`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.interactionDraft = {};
    showToast(t("answerAccepted"));
    await loadRuns();
    await loadSnapshot({ quiet: true });
    render();
  } finally {
    setControlBusy(control, false);
  }
}

async function startLegacyFollowup(runId, action, feedback, control) {
  setControlBusy(control, true);
  try {
    const run = await requestJson(`/api/runs/${encodeURIComponent(runId)}/followup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, feedback }),
    });
    state.selectedRunId = run.run_id;
    await loadRuns();
    await loadSnapshot();
    updateUrl();
    showToast(t("runStarted"));
    navigate("live", true);
  } finally {
    setControlBusy(control, false);
  }
}

async function stopRun(runId) {
  if (!runId) throw new Error(t("chooseRun"));
  if (!window.confirm(t("confirmStop"))) return;
  await requestJson(`/api/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
  await loadRuns();
  await loadSnapshot({ quiet: true });
  showToast(t("runStopped"));
  render();
}

async function selectRun(runId, view = state.activeView) {
  state.selectedRunId = runId || "";
  state.selectedGraphNode = null;
  state.focusNode = null;
  await loadSnapshot();
  updateUrl();
  navigate(view, true);
}

async function openArtifact(path) {
  if (!path) return;
  const cacheKey = `${state.selectedRunId || "root"}:${path}`;
  state.dialogReturnFocus = document.activeElement;
  refs.artifactTitle.textContent = artifactByPath(path)?.title || path.split("/").pop() || t("artifact");
  refs.artifactPath.textContent = path;
  refs.artifactBody.innerHTML = `<p class="meta">${t("loadingArtifact")}</p>`;
  if (!refs.artifactDialog.open) refs.artifactDialog.showModal();
  try {
    let artifact = state.artifactCache.get(cacheKey);
    if (!artifact) {
      const prefix = state.selectedRunId ? `/api/runs/${encodeURIComponent(state.selectedRunId)}/artifacts/` : "/api/artifacts/";
      artifact = await requestJson(prefix + encodePath(path));
      state.artifactCache.set(cacheKey, artifact);
    }
    if (!refs.artifactDialog.open || refs.artifactPath.textContent !== path) return;
    refs.artifactTitle.textContent = artifact.title || path;
    refs.artifactPath.textContent = `${artifact.path} · ${formatBytes(artifact.size_bytes)}`;
    refs.artifactBody.innerHTML = artifact.html
      ? `<article class="markdown-body">${sanitizeRenderedHtml(artifact.html)}</article>`
      : `<pre class="${artifact.kind === "json" ? "json-viewer" : "text-viewer"}">${escapeHtml(artifact.kind === "json" ? formatJson(artifact.text) : artifact.text || t("noPreview"))}</pre>`;
  } catch (error) {
    refs.artifactBody.innerHTML = emptyState("!", t("errorPrefix"), error.message || String(error));
    throw error;
  }
}

async function openRunLog(runId) {
  if (!runId) throw new Error(t("chooseRun"));
  stopLogPolling();
  state.activeLogRunId = runId;
  state.dialogReturnFocus = document.activeElement;
  refs.artifactTitle.textContent = t("log");
  refs.artifactPath.textContent = runId;
  refs.artifactBody.innerHTML = `<p class="meta">${t("loadingLog")}</p>`;
  if (!refs.artifactDialog.open) refs.artifactDialog.showModal();
  await refreshRunLog(runId);
  state.logTimer = window.setInterval(() => refreshRunLog(runId).catch(() => {}), 2200);
}

async function refreshRunLog(runId) {
  if (!refs.artifactDialog.open || state.activeLogRunId !== runId) return;
  const run = await requestJson(`/api/runs/${encodeURIComponent(runId)}`);
  const previous = refs.artifactBody.querySelector(".log-box");
  const nearBottom = !previous || previous.scrollHeight - previous.scrollTop - previous.clientHeight < 48;
  const previousTop = previous?.scrollTop || 0;
  refs.artifactTitle.textContent = run.run_name || t("log");
  refs.artifactPath.textContent = `${run.run_id} · ${statusLabel(run.status)}`;
  refs.artifactBody.innerHTML = `<pre class="log-box" tabindex="0">${escapeHtml(run.log_tail || t("logUnavailable"))}</pre>`;
  const current = refs.artifactBody.querySelector(".log-box");
  if (current) current.scrollTop = nearBottom ? current.scrollHeight : previousTop;
}

function stopLogPolling() {
  if (state.logTimer) window.clearInterval(state.logTimer);
  state.logTimer = null;
  state.activeLogRunId = null;
  const returnFocus = state.dialogReturnFocus;
  state.dialogReturnFocus = null;
  if (returnFocus instanceof HTMLElement && returnFocus.isConnected) window.setTimeout(() => returnFocus.focus(), 0);
}

async function mountInlineReport() {
  const container = document.querySelector("#inlineReport");
  if (!container || !artifactByPath("final_report.md")) return;
  const cacheKey = `${state.selectedRunId || "root"}:final_report.md`;
  try {
    let artifact = state.artifactCache.get(cacheKey);
    if (!artifact) {
      const prefix = state.selectedRunId ? `/api/runs/${encodeURIComponent(state.selectedRunId)}/artifacts/` : "/api/artifacts/";
      artifact = await requestJson(`${prefix}final_report.md`);
      state.artifactCache.set(cacheKey, artifact);
    }
    if (!container.isConnected) return;
    container.innerHTML = artifact.html ? sanitizeRenderedHtml(artifact.html) : `<pre class="text-viewer">${escapeHtml(artifact.text || "")}</pre>`;
    container.setAttribute("aria-busy", "false");
  } catch (error) {
    if (container.isConnected) container.innerHTML = emptyState("!", t("errorPrefix"), error.message || String(error));
  }
}

async function saveConfig(form) {
  const group = (state.config.groups || []).find((item) => item.id === form.dataset.group);
  const allowed = new Set(group?.fields || []);
  const values = {};
  state.configDirty.forEach((key) => {
    if (!allowed.has(key)) return;
    const field = state.config.fields?.[key];
    if (field?.secret && state.configDraft[key] === "") return;
    values[key] = state.configDraft[key];
  });
  if (!Object.keys(values).length) {
    state.configStatus = t("saved");
    state.configStatusKind = "success";
    renderViewOnly();
    return;
  }
  const submit = form.querySelector("button[type=submit]");
  setControlBusy(submit, true);
  state.configStatus = t("saving");
  state.configStatusKind = "";
  try {
    const previousConfig = state.config;
    state.config = await requestJson("/api/config", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    syncStudioDefaults(previousConfig, new Set(Object.keys(values)));
    state.configDraft = {};
    Object.values(state.config.fields || {}).forEach((field) => { state.configDraft[field.key] = field.value; });
    Object.keys(values).forEach((key) => state.configDirty.delete(key));
    const overridingVariables = Object.keys(values)
      .map((key) => state.config.fields?.[key])
      .filter((field) => field?.env_overridden)
      .map((field) => field.env_var);
    const message = overridingVariables.length
      ? t("savedWithOverrides", overridingVariables.join(", "))
      : t("saved");
    state.configStatus = message;
    state.configStatusKind = overridingVariables.length ? "warning" : "success";
    showToast(message);
    renderViewOnly();
  } catch (error) {
    state.configStatus = `Error: ${error.message || error}`;
    state.configStatusKind = "warning";
    renderViewOnly();
    throw error;
  } finally {
    setControlBusy(submit, false);
  }
}

async function reloadConfig() {
  state.configDraft = {};
  state.configDirty.clear();
  state.configStatus = "";
  state.configStatusKind = "";
  await loadConfig();
  state.configStatus = t("reloaded");
  state.configStatusKind = "success";
  renderViewOnly();
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, { ...options, headers: { Accept: "application/json", ...(options.headers || {}) } });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail || payload?.message || `HTTP ${response.status}`;
    throw new Error(String(detail));
  }
  return payload;
}

function buildEnhancedGraph() {
  const source = state.snapshot?.graph || { nodes: [], edges: [] };
  const allNodes = Array.isArray(source.nodes) ? source.nodes : [];
  const allEdges = Array.isArray(source.edges) ? source.edges : [];
  let focusIds = null;
  if (state.focusNode) {
    focusIds = new Set([state.focusNode]);
    allEdges.forEach((edge) => {
      if (edge.source === state.focusNode) focusIds.add(edge.target);
      if (edge.target === state.focusNode) focusIds.add(edge.source);
    });
  }
  const query = state.query;
  let nodes = allNodes.filter((node) => {
    if (focusIds && !focusIds.has(node.id)) return false;
    if (state.nodeKind && node.kind !== state.nodeKind) return false;
    if (state.confidence && node.kind === "claim" && String(node.group).toUpperCase() !== state.confidence) return false;
    return !query || graphNodeSearchText(node).includes(query);
  });
  nodes.sort((left, right) => {
    if (left.id === state.selectedGraphNode) return -1;
    if (right.id === state.selectedGraphNode) return 1;
    return `${left.kind}:${left.id}`.localeCompare(`${right.kind}:${right.id}`);
  });
  nodes = nodes.slice(0, 100);
  const ids = new Set(nodes.map((node) => node.id));
  const edges = allEdges.filter((edge) => ids.has(edge.source) && ids.has(edge.target));
  return { nodes, edges };
}

function graphProvenanceStats() {
  const graph = state.snapshot?.graph || { nodes: [], edges: [] };
  const papers = Array.isArray(state.snapshot?.papers) ? state.snapshot.papers : [];
  const claims = Array.isArray(state.snapshot?.claims) ? state.snapshot.claims : [];
  const gaps = Array.isArray(state.snapshot?.gaps)
    ? state.snapshot.gaps
    : (graph.nodes || []).filter((node) => node.kind === "gap");
  const supportEdges = (graph.edges || []).filter((edge) => edge.kind === "supports");
  const gapEdges = (graph.edges || []).filter((edge) => ["supports_gap", "partially_covers_gap", "challenges_gap"].includes(edge.kind));
  const mappedClaims = new Set(supportEdges.map((edge) => edge.target)).size;
  const mappedGaps = new Set(gapEdges.map((edge) => edge.target)).size;
  const usedPapers = new Set(supportEdges.map((edge) => edge.source)).size;
  return {
    mappedClaims,
    totalClaims: claims.length,
    unmappedClaims: Math.max(0, claims.length - mappedClaims),
    usedPapers,
    totalPapers: papers.length,
    unusedPapers: Math.max(0, papers.length - usedPapers),
    mappedGaps,
    totalGaps: gaps.length,
    unmappedGaps: Math.max(0, gaps.length - mappedGaps),
  };
}

function graphNodeSearchText(node) {
  const parts = [node.id, node.label, node.kind, node.group];
  const rawId = String(node.id || "").split(":").slice(1).join(":");
  if (node.kind === "paper") {
    const paper = (state.snapshot.papers || []).find((item) => item.paper_id === rawId);
    if (paper) parts.push(paper.title, paper.relevance, ...(paper.authors || []));
  } else if (node.kind === "claim") {
    const claim = (state.snapshot.claims || []).find((item) => item.citation_id === rawId);
    if (claim) parts.push(claim.claim, claim.source_title, claim.section);
  } else if (node.kind === "gap") {
    const gap = (state.snapshot.gaps || []).find((item) => item.gap_id === rawId);
    if (gap) parts.push(gap.title, gap.description, gap.gap_type, gap.why_it_matters, gap.potential_direction);
  } else if (node.kind === "idea") {
    const idea = (state.snapshot.ideas || []).find((item) => item.idea_id === rawId);
    if (idea) parts.push(idea.title, idea.one_liner, idea.description);
  }
  return parts.filter(Boolean).join(" ").toLocaleLowerCase();
}

function mountGraph() {
  const svg = document.querySelector("#researchGraph");
  const edgeLayer = document.querySelector("#graphEdges");
  const nodeLayer = document.querySelector("#graphNodes");
  if (!svg || !edgeLayer || !nodeLayer) return;
  const graph = buildEnhancedGraph();
  const anchors = { paper: 135, claim: 345, gap: 555, idea: 760, artifact: 830 };
  const counts = {};
  const nodes = graph.nodes.map((node) => {
    const index = counts[node.kind] || 0;
    counts[node.kind] = index + 1;
    const count = Math.max(1, graph.nodes.filter((item) => item.kind === node.kind).length);
    const jitter = (stableHash(node.id) % 31) - 15;
    return {
      ...node,
      x: (anchors[node.kind] || 450) + jitter,
      y: 68 + ((index + 1) / (count + 1)) * 500 + ((stableHash(`${node.id}:y`) % 23) - 11),
      vx: 0,
      vy: 0,
    };
  });
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const links = graph.edges.map((edge) => ({ ...edge, sourceNode: byId.get(edge.source), targetNode: byId.get(edge.target) })).filter((edge) => edge.sourceNode && edge.targetNode);
  const svgNs = "http://www.w3.org/2000/svg";
  const lineByEdge = [];
  links.forEach((edge) => {
    const line = document.createElementNS(svgNs, "line");
    line.setAttribute("class", `graph-edge ${edge.source === state.selectedGraphNode || edge.target === state.selectedGraphNode ? "focused" : ""}`);
    line.setAttribute("marker-end", "url(#graphArrow)");
    line.dataset.kind = edge.kind;
    edgeLayer.appendChild(line);
    lineByEdge.push([edge, line]);
  });
  const elementByNode = new Map();
  nodes.forEach((node) => {
    const group = document.createElementNS(svgNs, "g");
    group.setAttribute("class", `graph-node ${node.id === state.selectedGraphNode ? "selected" : ""}`);
    group.setAttribute("role", "button");
    group.setAttribute("tabindex", "0");
    group.setAttribute("aria-label", `${nodeKindLabel(node.kind)}: ${node.label}`);
    group.dataset.nodeId = node.id;
    const circle = document.createElementNS(svgNs, "circle");
    circle.setAttribute("r", node.id === state.focusNode ? "11" : "8.5");
    circle.setAttribute("fill", graphNodeColor(node.kind));
    const label = document.createElementNS(svgNs, "text");
    label.setAttribute("x", "13");
    label.setAttribute("y", "3.5");
    label.textContent = shortLabel(node.label, 24);
    group.append(circle, label);
    group.addEventListener("click", () => selectGraphNode(node.id));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectGraphNode(node.id);
      }
    });
    nodeLayer.appendChild(group);
    elementByNode.set(node.id, group);
  });
  const paint = () => {
    lineByEdge.forEach(([edge, line]) => {
      const dx = edge.targetNode.x - edge.sourceNode.x;
      const dy = edge.targetNode.y - edge.sourceNode.y;
      const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const ux = dx / distance;
      const uy = dy / distance;
      line.setAttribute("x1", (edge.sourceNode.x + ux * 10).toFixed(2));
      line.setAttribute("y1", (edge.sourceNode.y + uy * 10).toFixed(2));
      line.setAttribute("x2", (edge.targetNode.x - ux * 12).toFixed(2));
      line.setAttribute("y2", (edge.targetNode.y - uy * 12).toFixed(2));
    });
    nodes.forEach((node) => elementByNode.get(node.id)?.setAttribute("transform", `translate(${node.x.toFixed(2)} ${node.y.toFixed(2)})`));
  };
  const simulate = () => {
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const left = nodes[i];
        const right = nodes[j];
        let dx = right.x - left.x;
        let dy = right.y - left.y;
        let distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < 1) {
          dx = 1;
          dy = 0;
          distanceSquared = 1;
        }
        const force = Math.min(0.9, 780 / distanceSquared);
        const distance = Math.sqrt(distanceSquared);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        left.vx -= fx;
        left.vy -= fy;
        right.vx += fx;
        right.vy += fy;
      }
    }
    links.forEach((edge) => {
      const dx = edge.targetNode.x - edge.sourceNode.x;
      const dy = edge.targetNode.y - edge.sourceNode.y;
      const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const pull = (distance - 145) * 0.0017;
      const fx = (dx / distance) * pull;
      const fy = (dy / distance) * pull;
      edge.sourceNode.vx += fx;
      edge.sourceNode.vy += fy;
      edge.targetNode.vx -= fx;
      edge.targetNode.vy -= fy;
    });
    nodes.forEach((node) => {
      node.vx += ((anchors[node.kind] || 450) - node.x) * 0.006;
      node.vy += (320 - node.y) * 0.0007;
      node.vx *= 0.84;
      node.vy *= 0.84;
      node.x = Math.max(30, Math.min(870, node.x + node.vx));
      node.y = Math.max(36, Math.min(604, node.y + node.vy));
    });
  };
  paint();
  let frame = 0;
  const tick = () => {
    simulate();
    paint();
    frame += 1;
    if (frame < 54) state.graph.frame = window.requestAnimationFrame(tick);
    else state.graph.frame = null;
  };
  if (!prefersReducedMotion()) state.graph.frame = window.requestAnimationFrame(tick);
  applyGraphTransform();
  bindGraphPanZoom(svg);
}

function selectGraphNode(nodeId) {
  state.selectedGraphNode = nodeId;
  renderViewOnly();
  window.setTimeout(() => document.querySelector(`[data-node-id="${cssEscape(nodeId)}"]`)?.focus(), 0);
}

function bindGraphPanZoom(svg) {
  let pointerId = null;
  let origin = null;
  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".graph-node")) return;
    pointerId = event.pointerId;
    origin = { x: event.clientX, y: event.clientY, tx: state.graph.transform.x, ty: state.graph.transform.y };
    svg.setPointerCapture(pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (pointerId !== event.pointerId || !origin) return;
    const rect = svg.getBoundingClientRect();
    state.graph.transform.x = origin.tx + ((event.clientX - origin.x) * 900) / Math.max(1, rect.width);
    state.graph.transform.y = origin.ty + ((event.clientY - origin.y) * 640) / Math.max(1, rect.height);
    applyGraphTransform();
  });
  const end = (event) => {
    if (pointerId !== event.pointerId) return;
    pointerId = null;
    origin = null;
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomGraph(event.deltaY < 0 ? 1.1 : 1 / 1.1);
  }, { passive: false });
}

function zoomGraph(factor) {
  state.graph.transform.k = Math.max(0.55, Math.min(2.8, state.graph.transform.k * factor));
  applyGraphTransform();
}

function applyGraphTransform() {
  const canvas = document.querySelector("#graphCanvas");
  if (!canvas) return;
  const { x, y, k } = state.graph.transform;
  canvas.setAttribute("transform", `translate(${x} ${y}) scale(${k})`);
}

function cleanupGraph() {
  if (state.graph.frame) window.cancelAnimationFrame(state.graph.frame);
  state.graph.frame = null;
}

function graphNodeColor(kind) {
  return {
    paper: "var(--color-link)",
    claim: "var(--color-action)",
    gap: "var(--color-waiting)",
    idea: "var(--color-idea)",
    artifact: "var(--color-ink-tertiary)",
  }[kind] || "var(--color-ink-tertiary)";
}

function nodeKindLabel(kind) {
  return { paper: t("papers"), claim: t("claims"), gap: t("gaps"), idea: t("ideas"), artifact: t("artifacts") }[kind] || kind;
}

function edgeKindLabel(kind) {
  return {
    supports: t("relationshipSupports"),
    supports_gap: t("relationshipSupportsGap"),
    partially_covers_gap: t("relationshipPartialCoverage"),
    challenges_gap: t("relationshipChallengesGap"),
    evidence_for: t("relationshipEvidenceFor"),
    targets: t("relationshipTargets"),
  }[kind] || kind;
}

function stableHash(value) {
  let hash = 2166136261;
  for (const character of String(value)) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function selectedRun() {
  return state.runs.find((run) => run.run_id === state.selectedRunId) || null;
}

function artifactByPath(path) {
  return (state.snapshot?.artifacts || []).find((artifact) => artifact.path === path) || null;
}

function renderCompletionProof(completion = {}) {
  const checkpointCount = Array.isArray(completion.checkpoint_events) ? completion.checkpoint_events.length : 0;
  const auditState = completion.audit_passed === true ? "pass" : completion.audit_passed === false ? "fail" : "pending";
  const issues = Array.isArray(completion.audit_issues) ? completion.audit_issues : [];
  return `<div class="section-header"><div><p class="section-label">${t("completionProof")}</p><h2>${completion.verified ? t("verified") : t("notVerified")}</h2></div></div>
    <div class="completion-proof proof-stack">
      ${proofItem(t("allArtifacts"), completion.required_artifacts_ready ? t("verified") : t("missing"), completion.required_artifacts_ready ? "pass" : "pending")}
      ${proofItem(t("finalReportProof"), completion.final_report_present ? t("verified") : t("missing"), completion.final_report_present ? "pass" : "pending")}
      ${proofItem(t("gateProof"), completion.stage_12_gate_passed ? t("verified") : t("notVerified"), completion.stage_12_gate_passed ? "pass" : "pending")}
      ${proofItem(t("checkpointProof"), `${formatNumber(checkpointCount)} / 3`, checkpointCount === 3 ? "pass" : "pending")}
      ${proofItem(t("auditProof"), completion.audit_passed === true ? t("verified") : completion.audit_passed === false ? t("notVerified") : t("notChecked"), auditState)}
    </div>
    ${issues.length ? `<details class="advanced-options"><summary>${t("auditIssues")} · ${formatNumber(issues.length)}</summary><ul>${issues.slice(0, 20).map((issue) => `<li><strong>${escapeHtml(issue.code || issue.severity || "Issue")}</strong> ${escapeHtml(issue.message || "")}</li>`).join("")}</ul></details>` : ""}`;
}

function proofItem(label, value, status = "pending") {
  return `<div class="proof-item ${escapeAttr(status)}"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></div>`;
}

function metric(label, value) {
  return `<div class="metric"><strong>${escapeHtml(formatNumber(value))}</strong><span>${escapeHtml(label)}</span></div>`;
}

function fact(label, value) {
  const rendered = value == null || value === "" ? "—" : String(value);
  return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(rendered)}</dd></div>`;
}

function modeRadio(value, label, checked, disabled) {
  return `<label><input type="radio" name="mode" value="${escapeAttr(value)}" ${checked ? "checked" : ""} ${disabled ? "disabled" : ""} /><span>${escapeHtml(label)}</span></label>`;
}

function buttonHtml(action, label, className = "button-secondary") {
  return `<button class="button ${escapeAttr(className)}" type="button" data-action="${escapeAttr(action)}">${escapeHtml(label)}</button>`;
}

function artifactRow(artifact) {
  return `<button class="artifact-row" type="button" data-action="open-artifact" data-path="${escapeAttr(artifact.path)}"><span><strong>${escapeHtml(artifact.title)}</strong><small>${escapeHtml(artifact.path)}</small></span><span class="artifact-kind">${escapeHtml(artifact.kind)}</span></button>`;
}

function icon(name) {
  const paths = {
    plus: '<path d="M10 4v12M4 10h12" />',
    minus: '<path d="M4 10h12" />',
    arrow: '<path d="M4 10h12m-4-4 4 4-4 4" />',
    pulse: '<path d="M2 10h4l2-5 4 10 2-5h4" />',
    external: '<path d="M8 4h8v8M16 4l-7 7" /><path d="M14 11v5H4V6h5" />',
    reset: '<path d="M5 6v5h5" /><path d="M6 10a6 6 0 1 1 1 5" />',
  };
  return `<svg viewBox="0 0 20 20" aria-hidden="true">${paths[name] || paths.arrow}</svg>`;
}

function emptyState(mark, title, body, action = "") {
  return `<section class="empty-state"><span class="empty-mark" aria-hidden="true">${escapeHtml(mark)}</span><h2>${escapeHtml(title)}</h2><p>${escapeHtml(body)}</p>${action}</section>`;
}

function statusLabel(status) {
  const key = {
    queued: "queued",
    running: "running",
    waiting_for_input: "waitingForInput",
    pipeline_completed: "pipelineCompleted",
    completed: "complete",
    complete: "complete",
    failed: "failed",
    stopped: "stopped",
    stale: "stale",
    checkpoint_reached: "checkpointReached",
    pending: "pending",
  }[status];
  return key ? t(key) : status ? String(status).replaceAll("_", " ") : t("unknown");
}

function stageLabel(stageId) {
  if (!stageId || stageId === "complete") return stageId === "complete" ? t("complete") : t("notStarted");
  const stage = normalizePipelineStages(state.snapshot?.pipeline || {}).find((item) => item.id === stageId);
  return stage ? `${stage.number || stageNumber(stageId)} · ${localizedStageName(stageId, stage.name)}` : String(stageId).replace("stage_", "Stage ");
}

function localizedStageName(stageId, fallback = "") {
  const names = state.language === "zh" ? {
    stage_1: "研究需求定义",
    stage_2: "研究问题形式化",
    stage_3: "文献检索与综述",
    "stage_3.5": "论文全文精读",
    stage_4: "论文定位分析",
    stage_5: "线索驱动扩展",
    stage_6: "论据与来源绑定",
    stage_7: "知识综合与研究空白",
    stage_8: "设计空间构建",
    stage_9: "候选想法生成",
    "stage_9.5": "Elo 锦标赛排名",
    stage_10: "对抗性辩论",
    stage_11: "可行性评估",
    stage_12: "最终研究报告",
  } : {
    stage_1: "Requirement intake",
    stage_2: "Task formalization",
    stage_3: "Literature survey",
    "stage_3.5": "Paper deep reading",
    stage_4: "Position analysis",
    stage_5: "Hook-driven expansion",
    stage_6: "Evidence binding",
    stage_7: "Knowledge synthesis",
    stage_8: "Design space",
    stage_9: "Idea generation",
    "stage_9.5": "Elo tournament",
    stage_10: "Adversarial debate",
    stage_11: "Feasibility assessment",
    stage_12: "Final report",
  };
  return names[stageId] || fallback || String(stageId || "").replace("stage_", "Stage ");
}

function progressActivity(progress) {
  if (!progress?.stage) return "";
  const phaseKey = `phase_${progress.phase || ""}`;
  const translated = t(phaseKey);
  const phase = translated === phaseKey
    ? String(progress.phase || "").replaceAll("_", " ")
    : translated;
  return progress.subject ? `${phase} · ${progress.subject}` : phase;
}

function renderStageProgress(progress) {
  if (!progress?.stage) return "";
  const current = Number(progress.current);
  const total = Number(progress.total);
  const determinate = progress.indeterminate === false && Number.isFinite(current) && Number.isFinite(total) && total > 0;
  const percent = determinate ? Math.max(0, Math.min(100, Number(progress.percent) || 0)) : 0;
  const unitKey = `unit_${progress.unit || ""}`;
  const unitText = t(unitKey) === unitKey ? String(progress.unit || "") : t(unitKey);
  const counts = progress.counts || {};
  const metrics = [];
  if (Number(counts.batches_total) > 0) metrics.push([t("metric_batches"), `${formatNumber(counts.batches_completed || 0)} / ${formatNumber(counts.batches_total)}`]);
  ["full_text", "failed", "claims", "gaps", "evidence_links", "axes", "combinations", "comparisons"].forEach((key) => {
    if (Object.hasOwn(counts, key) && (Number(counts[key]) > 0 || ["full_text", "failed"].includes(key))) {
      metrics.push([t(`metric_${key}`), formatNumber(counts[key])]);
    }
  });
  if (Number(counts.round_target) > 0) metrics.push([t("metric_debate_rounds"), `${formatNumber(counts.debate_rounds || 0)} / ${formatNumber(counts.round_target)}`]);
  const label = `${t("stageProgress")}: ${localizedStageName(progress.stage, progress.name)}`;
  const stageStatus = progress.status || "running";
  return `<section class="stage-progress-card ${escapeAttr(stageStatus)}" aria-label="${escapeAttr(label)}">
    <div class="stage-progress-heading">
      <div><p class="section-label">${t("currentActivity")}</p><h3>${escapeHtml(`${progress.number || stageNumber(progress.stage)} · ${localizedStageName(progress.stage, progress.name)}`)}</h3><p>${escapeHtml(progressActivity(progress))}</p></div>
      <div class="stage-progress-summary">
        ${determinate ? `<div class="stage-progress-measure"><strong>${formatNumber(current)} / ${formatNumber(total)}</strong><span>${escapeHtml(unitText)}</span></div>` : ""}
        <span class="status-pill ${escapeAttr(stageStatus)}">${escapeHtml(statusLabel(stageStatus))}</span>
      </div>
    </div>
    ${determinate ? `<div class="stage-subprogress" role="progressbar" aria-label="${escapeAttr(label)}" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100" aria-valuetext="${escapeAttr(`${formatNumber(current)} / ${formatNumber(total)} ${unitText}`)}"><span style="--progress:${percent}%"></span></div>` : ""}
    ${metrics.length ? `<dl class="stage-progress-metrics">${metrics.map(([name, value]) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>` : ""}
    ${progress.updated_at ? `<p class="stage-progress-updated">${t("lastActivity")} · <time datetime="${escapeAttr(progress.updated_at)}" data-relative-time="${escapeAttr(progress.updated_at)}" title="${escapeAttr(formatDate(progress.updated_at))}">${escapeHtml(formatRelativeTime(progress.updated_at))}</time></p>` : ""}
  </section>`;
}

function traceStageLabel(stageId, fallback) {
  const labels = state.language === "zh" ? {
    stage_1: "需求",
    stage_2: "定义",
    stage_3: "综述",
    "stage_3.5": "精读",
    stage_4: "定位",
    stage_5: "扩展",
    stage_6: "证据",
    stage_7: "综合",
    stage_8: "空间",
    stage_9: "想法",
    "stage_9.5": "排名",
    stage_10: "辩论",
    stage_11: "可行性",
    stage_12: "报告",
  } : {
    stage_1: "Intake",
    stage_2: "Formalize",
    stage_3: "Survey",
    "stage_3.5": "Deep read",
    stage_4: "Position",
    stage_5: "Expand",
    stage_6: "Evidence",
    stage_7: "Synthesis",
    stage_8: "Design",
    stage_9: "Ideas",
    "stage_9.5": "Elo",
    stage_10: "Debate",
    stage_11: "Feasibility",
    stage_12: "Report",
  };
  return labels[stageId] || shortLabel(fallback, 11);
}

function stageNumber(stageId) {
  const value = String(stageId || "").replace("stage_", "");
  if (!value) return "—";
  return value.includes(".") ? value.padStart(4, "0") : value.padStart(2, "0");
}

function formatNumber(value, maximumFractionDigits = 0) {
  if (typeof value === "string" && value.trim().endsWith("%")) return value;
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value ?? "0");
  return new Intl.NumberFormat(state.language === "zh" ? "zh-CN" : "en-US", { maximumFractionDigits }).format(numeric);
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(state.language === "zh" ? "zh-CN" : "en-US", { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function formatRelativeTime(value, now = Date.now()) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value || "—");
  const elapsedSeconds = Math.max(0, Math.floor((now - date.getTime()) / 1000));
  if (elapsedSeconds < 10) return t("activityNow");
  if (elapsedSeconds < 60) return t("activitySecondsAgo", elapsedSeconds);
  const elapsedMinutes = Math.floor(elapsedSeconds / 60);
  if (elapsedMinutes < 60) return t("activityMinutesAgo", elapsedMinutes);
  const elapsedHours = Math.floor(elapsedMinutes / 60);
  if (elapsedHours < 24) return t("activityHoursAgo", elapsedHours);
  return t("activityDaysAgo", Math.floor(elapsedHours / 24));
}

function updateRelativeTimes() {
  document.querySelectorAll("[data-relative-time]").forEach((element) => {
    const value = element.dataset.relativeTime || "";
    const nextText = formatRelativeTime(value);
    if (element.textContent !== nextText) element.textContent = nextText;
    element.title = formatDate(value);
  });
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1024) return `${formatNumber(bytes)} B`;
  const units = ["KB", "MB", "GB"];
  let amount = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && amount >= 1024; index += 1) {
    amount /= 1024;
    unit = units[index];
  }
  return `${formatNumber(amount, amount < 10 ? 1 : 0)} ${unit}`;
}

function formatJson(text) {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return String(text || "");
  }
}

function formatConfigValue(value) {
  if (value === true) return "true";
  if (value === false) return "false";
  if (value == null || value === "") return "—";
  return String(value);
}

function tailText(value, limit) {
  const text = String(value || "");
  return text.length > limit ? `…\n${text.slice(-limit)}` : text;
}

function shortLabel(value, limit = 36) {
  const text = String(value || "").trim();
  return text.length > limit ? `${text.slice(0, Math.max(1, limit - 1)).trimEnd()}…` : text;
}

function matchesQuery(value, query) {
  if (!query) return true;
  try {
    return JSON.stringify(value).toLocaleLowerCase().includes(query);
  } catch {
    return String(value).toLocaleLowerCase().includes(query);
  }
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function encodePath(path) {
  return String(path).split("/").map((part) => encodeURIComponent(part)).join("/");
}

function safeHref(value) {
  const href = String(value || "").trim();
  if (!href) return "";
  const compact = href.replace(/[\u0000-\u0020]+/g, "");
  if (/^\/\//.test(compact)) return "";
  const scheme = compact.match(/^([a-z][a-z0-9+.-]*):/i);
  if (scheme && !["http", "https", "mailto"].includes(scheme[1].toLowerCase())) return "";
  return href;
}

function sanitizeRenderedHtml(rendered, headingOffset = 1) {
  const template = document.createElement("template");
  template.innerHTML = String(rendered || "");
  template.content.querySelectorAll("script, style, iframe, object, embed, link, meta, base, form, input, button").forEach((node) => node.remove());
  if (headingOffset > 0) {
    template.content.querySelectorAll("h1, h2, h3, h4, h5, h6").forEach((heading) => {
      const level = Math.min(6, Number(heading.tagName.slice(1)) + headingOffset);
      const replacement = document.createElement(`h${level}`);
      Array.from(heading.attributes).forEach((attribute) => replacement.setAttribute(attribute.name, attribute.value));
      while (heading.firstChild) replacement.append(heading.firstChild);
      heading.replaceWith(replacement);
    });
  }
  template.content.querySelectorAll("*").forEach((node) => {
    Array.from(node.attributes).forEach((attribute) => {
      const name = attribute.name.toLowerCase();
      if (name.startsWith("on") || ["srcdoc", "formaction", "style"].includes(name)) node.removeAttribute(attribute.name);
      if (["href", "src", "xlink:href"].includes(name)) {
        const safe = safeHref(attribute.value);
        if (safe) node.setAttribute(attribute.name, safe);
        else node.removeAttribute(attribute.name);
      }
    });
    if (node.tagName === "A") {
      node.setAttribute("rel", "noopener noreferrer");
      if (/^https?:/i.test(node.getAttribute("href") || "")) node.setAttribute("target", "_blank");
    }
  });
  return template.innerHTML;
}

function setStatusKey(key, ...args) {
  state.globalStatus = { key, args, detail: "" };
  renderGlobalStatus();
}

function setErrorStatus(detail) {
  state.globalStatus = { key: "errorPrefix", args: [], detail };
  renderGlobalStatus();
}

function renderGlobalStatus() {
  const status = state.globalStatus || {};
  const message = status.key ? t(status.key, ...(status.args || [])) : "";
  refs.status.textContent = status.detail ? `${message}: ${status.detail}` : message;
}

function showToast(message) {
  if (state.toastTimer) window.clearTimeout(state.toastTimer);
  refs.toast.textContent = message;
  refs.toast.classList.add("visible");
  state.toastTimer = window.setTimeout(() => refs.toast.classList.remove("visible"), 3600);
}

function handleError(error) {
  const message = error?.message || String(error || t("unknown"));
  setErrorStatus(message);
  showToast(`${t("errorPrefix")}: ${message}`);
}

function applyLanguage() {
  document.documentElement.lang = state.language === "zh" ? "zh-CN" : "en";
  document.documentElement.dataset.language = state.language;
  document.title = state.language === "zh" ? "AutoIdea 研究观测台" : "AutoIdea Research Observatory";
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    if (key) element.textContent = t(key);
  });
  refs.languageToggle.textContent = state.language === "zh" ? "EN" : "中文";
  refs.languageToggle.title = t("switchLanguage");
  refs.languageToggle.setAttribute("aria-label", t("switchLanguage"));
  refs.refreshButton.title = t("refresh");
  refs.refreshButton.setAttribute("aria-label", t("refresh"));
  refs.navToggle.setAttribute("aria-label", state.navOpen ? t("closeNavigation") : t("openNavigation"));
}

function t(key, ...args) {
  const value = I18N[state.language]?.[key] ?? I18N.en[key] ?? key;
  return typeof value === "function" ? value(...args) : value;
}

function getInitialLanguage() {
  const stored = safeStorageGet("autoidea-language");
  if (stored === "en" || stored === "zh") return stored;
  return String(navigator.language || "en").toLowerCase().startsWith("zh") ? "zh" : "en";
}

function safeStorageGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeStorageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Storage may be disabled; language still works for this page session.
  }
}

function initialParam(key) {
  try {
    return new URL(window.location.href).searchParams.get(key) || "";
  } catch {
    return "";
  }
}

function updateUrl(push = false) {
  const url = new URL(window.location.href);
  if (state.activeView && state.activeView !== "studio") url.searchParams.set("view", state.activeView);
  else url.searchParams.delete("view");
  if (state.selectedRunId) url.searchParams.set("run", state.selectedRunId);
  else url.searchParams.delete("run");
  if (state.query) url.searchParams.set("q", state.query);
  else url.searchParams.delete("q");
  if (state.confidence) url.searchParams.set("confidence", state.confidence);
  else url.searchParams.delete("confidence");
  if (state.nodeKind) url.searchParams.set("node", state.nodeKind);
  else url.searchParams.delete("node");
  if (state.activeConfigGroup && state.activeConfigGroup !== "quick") url.searchParams.set("group", state.activeConfigGroup);
  else url.searchParams.delete("group");
  const method = push ? "pushState" : "replaceState";
  window.history[method]({ view: state.activeView, run: state.selectedRunId }, "", url);
}

function navigate(view, push = false) {
  if (!VIEW_META[view]) view = "studio";
  const changed = state.activeView !== view;
  state.activeView = view;
  setNavOpen(false);
  updateUrl(push && changed);
  render();
  if (changed) {
    window.scrollTo({ top: 0, behavior: "auto" });
    window.setTimeout(() => refs.main.focus({ preventScroll: true }), 0);
  }
}

function setNavOpen(open) {
  const wasOpen = state.navOpen;
  state.navOpen = Boolean(open);
  refs.nav.classList.toggle("open", state.navOpen);
  refs.navToggle.setAttribute("aria-expanded", String(state.navOpen));
  refs.navToggle.setAttribute("aria-label", state.navOpen ? t("closeNavigation") : t("openNavigation"));
  refs.navScrim.hidden = !state.navOpen;
  if (wasOpen && !state.navOpen) refs.navToggle.focus();
}

function isEditingView() {
  if (refs.artifactDialog.open || state.activeView === "settings" || state.activeView === "map") return true;
  const active = document.activeElement;
  return Boolean(active && refs.viewRoot.contains(active) && active.matches("input, textarea, select, button, summary, [contenteditable=true]"));
}

function setControlBusy(control, busy) {
  if (!(control instanceof HTMLElement)) return;
  if ("disabled" in control) control.disabled = Boolean(busy);
  if (busy) control.setAttribute("aria-busy", "true");
  else control.removeAttribute("aria-busy");
}

function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches || false;
}

function cssEscape(value) {
  if (window.CSS?.escape) return window.CSS.escape(String(value || ""));
  return String(value || "").replace(/[^a-zA-Z0-9_-]/g, (character) => `\\${character}`);
}
