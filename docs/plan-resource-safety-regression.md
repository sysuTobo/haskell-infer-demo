# 资源安全与回归门禁修复计划

状态：**已实施并验证（2026-09-23）**。用户授权后按本计划完成；下面的步骤、文件清单与验收标准保留原样，实施与验收结果见文末「执行结果」。

---

## 执行结果（2026-09-23）

**提交**（均在 pod 上验证后推送到 `main`）：`6d6e306`（资源安全与回归门禁主体：生成预算/EOS/错误语义、Tokenizer 容量协议与增量解码、safetensors 有界解析、分配即时归属、跨卡读完成依赖、MLA 共享内存上限、TP 分片覆盖规则、门禁汇总）、`6417c72`（collective 元素类型跟随数据）、`fe2c2d4`（EP 合并改为 FP32、只舍入一次）。

**sm_86（2× A40）实测通过**：`bash scripts/build.sh --tests` → cargo 11/11、ctest 11/11（含 `test_mla`、`test_collective` 双卡、`test_engine_resources`、`test_safetensors`）、`infer-tests` 41/41（含 `INFER_MODEL_DIR` 快照）、`infer-generation-tests` 14/14；Qwen3.8 golden 与重构前基线**逐位相同**（`max_abs == 0`）；TP2 等价 rms ≤0.05 且 greedy token 全同；DeepSeek-V2-Lite / Qwen3-4B / 合成 Qwen3-Next / Qwen3-30B-A3B 四族 engine 与边界回归通过；真实 CLI 的零预算、负预算、长文本与 UTF-8 streaming 行为正确。

**期间发现并修复**：pod 构建暴露出本计划引入的一个回归——Qwen3.8-27B 在 1198 个文本张量之外还带一个 **rank-5 的 Conv3D vision 张量**，严格 rank 上限曾导致整个目录拒收、模型不可加载；已把上限改为解析边界（8）并补 CPU 用例固定该行为。

**遗留**：EP 放置等价在 FP32 合并落地后的复跑按要求中止，结论待定（门禁不变：`test_tp.py --ep 2`，token 全同 + rms ≤0.05）；合并前实测为 0.05–0.22，族带 0.1–0.6（同模型对独立 oracle 为 0.09–0.31）。sm_90a 仍只有编译/产物验证（无 H200）。本机（无 nvcc）完成的部分：Haskell 全量类型检查与链接、两个 hspec suite（HUnit 不可拉取，用等价 harness 实跑）、两个 C CPU 套件（含 ASan/UBSan）、Rust 测试、所有改动 CUDA 文件的 host 编译器语法检查。

## Context

用户要求修复上一轮代码评审发现的问题，而非继续讨论或实现训练方案。已复现：`tests/Spec.hs` 的基础 fixture 缺少四个严格 MLA 字段；真实 Haskell Generation 模块在首 token EOS、零生成预算下继续 decode。静态确认：跨卡广播缺接收完成到源缓冲复用的依赖；safetensors 文件长度与目标分配脱节；初始化失败时部分分配尚未登记 owned；Tokenizer 会静默截断并独立解码字节片段。另补 MLA 长上下文共享内存限制的提前校验。

目标是保持合法模型的数值计算不变，修复失败路径及边界行为，并将 CPU、Rust、Haskell、GPU 回归接入现有测试入口。不改训练/GSPO/异步 RL、不重写 attention、不改 AGENTS.md 或 CI。

## 工作约束

- 本机仓库零 commit 且已有大量改动：不创建 worktree，不提交、不覆盖其他会话文件。
- 实施前检查当前文件与权威 pod 对应文件的状态/摘要；只同步本次明确路径。有分歧先读差异，不进行整目录覆盖。
- 本机测试产物用独立临时目录；不覆盖现有 CUDA/tokenizer 桩库。
- GPU 验证采用项目既有 pod 流程及现存权重/参考，不重新下载模型。不假定 pod 或工具链已就绪。
- 本任务不自行发布到 GitHub；保留清晰可审查的工作树变更。

## 1. 先恢复并扩大 Haskell 回归

修改 `tests/Spec.hs`、`src/Infer/Generation.hs`、`src/Main.hs`、`src/Infer/Runtime.hs`、`haskell-infer-demo.cabal`。
新增独立 CPU 生成回归入口 `tests/GenerationSpec.hs` 与仅该测试链接的 `tests/generation_engine_stub.c`，复用上一轮实际模块加 C mock 的验证方式，不引入通用推理引擎抽象。

- 补齐基础 fixture 的四个 MLA 字段，非 MLA 模型设零。
- 先加入失败用例：首 EOS、零预算、负预算、正常计数、后续 EOS、prefill/decode 错误；同时覆盖流式和非流式。
- 生成预算为零不调用引擎；负数由 CLI 拒绝，内部入口也明确处理；首 token 与后续 token 使用一致的停止条件。
- 不把 engine 错误转换成正常的空/部分输出并成功退出；错误上抛，由入口报告失败。
- `bracket` 管理 Runtime，初始化 tokenizer 成功但 engine 失败时释放 tokenizer；异常/输出错误同样保证清理。

## 2. Tokenizer 无损容量协议与增量解码

修改 `tokenizer-ffi/src/lib.rs`、`src/Infer/Tokenizer.hs`、`src/Infer/Generation.hs`，必要时仅调整 Cabal 测试链接配置。

- encode/decode 增加长度查询语义，空输出缓冲合法并返回完整所需长度；容量不足返回明确错误，不部分写入。文本容量计入结尾 NUL，长度返回值明确是否包含 NUL。
- Haskell 动态分配所需空间，显式 UTF-8 编解码并使用返回字节长度，不能把错误变成 `[]`/空字符串。
- 使用已缓存的 tokenizers 0.20.4 `step_decode_stream`，新增拥有 tokenizer clone 和解码状态的独立 stream handle；不使用自引用 Rust 借用或延长伪造生命周期。
- 分开 feed 与 pending-output 查询/drain：一次 token 只推进一次状态；查长和容量重试不会二次 feed，成功复制才清 pending。
- reset 清理全部状态，finish 明确处理未完成字节片段且幂等；streaming 与整段 decode 使用同一 special-token 策略。通过 bracket 释放 stream handle。
- Rust 内存构造 BPE/ByteFallback fixture，不下载模型；覆盖跨 token UTF-8、中英文、空/特殊 token、超过旧缓冲限制、容量不足不修改缓冲、重试、finish/reset。
- Haskell 测试继续使用真实 Generation/Tokenizer 包装层；Rust ABI 与增量状态由 Rust 测试及一个轻量 CPU 集成测试共同覆盖。

## 3. Safetensors 边界校验与独立 CPU 测试

修改 `csrc/safetensors_loader.cu`、`csrc/include/layers.h`、`csrc/engine.cu`、`csrc/CMakeLists.txt`。
新增 `csrc/include/safetensors.h`、`csrc/safetensors.cpp`、`tests/safetensors_test.cpp`，将不依赖 CUDA 的元数据解析/文件边界验证单独编译；统一重复的 TensorInfo 定义。

- 使用有边界的 safetensors schema 解析，拒绝非法/重复字段、未知 dtype、错误整数、缺失 shape/offset、无进展输入；不依赖 strstr 推测对象范围。
- header 长度先与实际文件大小及合理上限比较，再分配；所有读/seek 都检查结果；FILE/DIR 使用 RAII。
- checked arithmetic 计算 shape×dtype 字节数，校验维度范围、offset 非负/有序、数据区范围和精确长度；错误不被 scan_dir 忽略，不返回部分成功索引。
- whole/rows/slice 上传都显式接收目标容量，校验行/列/gather 索引及乘法溢出，重新核对文件读取边界，传播 CUDA 错误。
- CPU 小文件 fixture 覆盖合法 BF16/F32/F16、短 header、长度过大、字节数偏大/偏小、负数/反向/越界 offset、溢出、重复键、坏 shard、合法分片和行重排。
- 目标为不改变合法权重的字节顺序和读取结果；不扩展量化格式。

## 4. 初始化分配即时归属

修改 `csrc/engine.cu`，复用 `LayerWeights::owned` 与已有 destroy 排空所有 stream 的机制，不重构整个引擎。

- 所有 layer-owned 分配在任何可能失败的上传/初始化前登记；短生命周期设备分配 guard 保护分配到登记之间的异常，包括 vector push 失败。
- 覆盖 load_role/view/rows、router/expert/shared experts、融合 GDN 各视图、KV/MLA cache、GDN norm/conv/SSM state。
- ctx-owned 指针保持已有唯一归属，避免重复登记/双重释放；guard 清理必须选择所属 device，析构不抛异常。
- 新增 `tests/test_engine_resources.py`，在临时目录生成小型 checkpoint，故意缺少晚期普通/专家张量或提供错误 shape；反复创建失败，确认 handle 为空、错误可解释、资源回收后仍可创建合法模型。
- CUDA context/cuBLAS 预热后测每卡显存使用，区分一次性缓存和持续泄漏。不用大模型 OOM 作为测试手段。

## 5. 跨卡广播读完成依赖

修改 `csrc/collective.cu` 与 `tests/kernels/test_collective.cu`；必要时更新 `layers.h` 中内部 collective 契约。

- 保留 producer event→consumer wait 的生产依赖。
- 每个接收 stream 在广播复制之后记录完成事件，leader 在复用源缓冲之前等待这些事件。事件记录/等待按代际提交，避免复用 producer event 导致循环等待。
- 不用全局 cudaDeviceSynchronize 取代正确的 stream 依赖。
- 增加连续 all-reduce、紧接着 leader 覆写、延迟接收端/完成前读取校验的 GPU 用例；轮间不插 host 同步把竞态掩盖。
- 有 P2P 时验证真正异步路径，无 P2P 时验证驱动 staging 路径；没有两卡必须明确 SKIP，不当作双卡验证通过。

## 6. MLA 共享内存容量提前拒绝

修改 `csrc/kernels/mla.cu`、`csrc/include/layers.h`、`csrc/engine.cu`、`tests/kernels/test_mla.cu`。

- 根据当前设备/kernel 的默认每 block 共享内存限额，扣除静态归约区，计算本实现支持的最大序列长度。
- engine 初始化在大型分配前拒绝超过此上限的配置；直接 kernel 入口也进行容量检查并返回明确错误。
- 测支持上限及上限+1、16K 不支持路径，保留65/72/140 token历史修复回归。
- 此次不引入 opt-in 大共享内存或新 softmax 算法，不声称新增长上下文支持。

## 7. 汇总门禁与验收

修改 `scripts/build.sh`、`csrc/CMakeLists.txt`、`haskell-infer-demo.cabal`，只扩充现有测试组织：`--tests` 执行 cargo test、全部 Cabal test suites（带 --enable-tests）、CTest，失败立即返回非零。不修改现有数值容差来让测试通过。

本机：
- 独立目录编译并运行 C descriptor、CPU safetensors 测试；可用 ASan/UBSan 检查畸形输入。
- Rust `cargo test --locked --offline` 优先，使用独立 CARGO_TARGET_DIR 避免覆盖本机桩库。
- Haskell 应用全量类型检查；运行无需 CUDA 的 Generation mock suite。
- 完整 Hspec 若受 HUnit 缺包阻挡，在 pod 完成，不把抽取 fixture 类型检查冒充全套测试。

Pod：
- 同步前后逐文件摘要检查，按正确 mtime 构建。
- `bash scripts/build.sh --tests` 全门禁，包括新增测试。
- 双卡 collective 压力与 engine 初始化失败资源测试。
- 现存小模型/合成模型、DeepSeek MLA 的 engine 与边界回归；TP/EP 回归验证通信改动。
- 用现有 Qwen3.8 golden 验证合法推理数值不变；独立参考保持原有 top-1 和各族 RMS 标准。
- 实际 CLI 检查零预算、EOS、长文本输出和 UTF-8 streaming；无 GPU 仅报告 CPU 检查，不声称完整修复已验证。

最终独立 review 实际 diff 和测试结果，修复新发现问题后复测；向用户报告改动、通过的门禁及仍被环境阻挡的项目。
