# FlagGems-Experimental 仓库开发策略

## 背景介绍

KernelGen 提交 PR 的频率非常高。为了解决算子库 PR Review 速度慢的问题，将算子生成 与提交 PR 等流程解耦，建立了 FlagGems-Experimental 仓库。FlagGems-Experimental 仓库不强制 Review 但是需要支持 CI/CD。

## 开发策略

FlagGems-Experimental 仓库的最终目标是日后合入 FlagGems 官方仓库，因此需要按照“上游优先（Upstream-First）”的规范来组织分支：CI 的改动与代码逻辑分家，只有在需要跑自动化测试时才发生交集。

### 结构设计

- **`master` 分支**：只同步不修改，代码必须与 `upstream/master` 100% 相同。

- **`infra-ci`分支**：从 `master` 签出，只修改`.github/workflows/`或测试脚本。

- **`feat/kernelgen` 分支**：从干净的 `master`签出，合入所有算子修改，作为 Experimental 仓库的默认分支。

- **`feat/op_xxx`分支**：从干净的 `master`签出，只写算子内核。


### 提交流程

开发算子时从`master`签出 `feat/op_xxx`分支。向`infra-ci`分支提 PR，CI 通过后同步合入`feat/kernelgen`。

### 同步脚本

编写一个专门的自动化 Workflow。当且仅当一个分支向`infra-ci` 提的 PR 被合并（Merge）时，自动通过 GitHub Actions 将该代码**同步合并**到 `feat/kernelgen`。

`infra-ci` 分支中，添加一个自动化同步脚本`.github/workflows/sync-to-kernelgen.yaml`。
