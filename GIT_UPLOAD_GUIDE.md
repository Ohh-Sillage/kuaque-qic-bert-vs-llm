# GIT_UPLOAD_GUIDE.md — Git 历史清理与仓库上传指南

> 适用场景：本地提交历史中混入了大文件 / 无用提交，需要清空历史并推送到（新的）GitHub 仓库
> 操作环境：Windows + cmd（命令提示符）/ PyCharm
> 本文以本项目 2026-10 实操为蓝本整理，命令均已实际验证

---

## 一、背景：为什么需要清理

本项目初次提交时误将以下内容提交进了 git：

| 误提交内容 | 体积 | 问题 |
|---|---|---|
| `outputs/checkpoints/*.pt`（3 个模型检查点） | 共 1.17 GB | 超出 GitHub 单文件 100 MB 硬性限制，**必然推送失败** |
| `src/`、`src_llm/` 的 `__pycache__/*.pyc` | 共 12 个文件 | 本地编译缓存，无版本价值 |
| `.idea/` 目录 | — | IDE 本地配置（含本机路径等个人信息） |

后果：本地 `.git` 对象库膨胀到 **1.08 GiB**，且该提交尚未推送——**此时重写历史代价最低**。

### 诊断命令（清理前先体检）

```bat
:: 查看仓库对象库大小（清理前 1.08 GiB → 清理后应 < 100 MB）
git count-objects -vH

:: 查看被跟踪文件里有哪些"不该提交"的内容
git ls-files | findstr /i /c:".pt" /c:"pycache" /c:".idea"

:: 查看提交历史（本项目清理前仅 1 条 first commit，却混入 1.17GB 大文件）
git log --oneline

:: 查看本地分支与上游的跟踪关系（无 [origin/main] 说明从未推送成功）
git branch -vv

:: 查看远程仓库现有引用（空输出 = 远程为空，可直接推送无需 force）
git ls-remote origin
```

---

## 二、总体思路

```
删除 .git（清空全部历史与大文件对象）
    → git init 重建
    → .gitignore 排除大文件 / 缓存 / IDE 配置
    → 一次干净的初始提交
    → 推送到远程仓库
```

- 远程仓库已存在且为空 → 直接 `git push`；
- 远程仓库已有内容且不冲突需求 → 需 `--force` 覆盖（慎用）；
- 远程未配置 → 先在 GitHub 建空仓库，再 `git remote add origin`。

---

## 三、详细步骤

### 步骤 0 · 准备工作

1. **关闭 PyCharm**（避免 .git 内文件被占用，导致删除失败）
2. 确认 git 可用：

   ```bat
   git --version
   ```

3. 确认远程仓库已在 GitHub 创建好，拿到地址（本项目：`https://github.com/Ohh-Sillage/kuaque-qic-bert-vs-llm.git`）
4. （可选保险）把整个项目文件夹复制一份作备份；旧历史若曾推送到别的远程仓库，随时可 clone 找回

### 步骤 1 · 清理旧历史

```bat
cd /d C:\Users\Latecomer\Desktop\共享\Project\kuaque-qic-bert-vs-llm
rmdir /s /q .git
```

- 无任何输出即成功；`dir /a` 确认 `.git` 已消失
- 等价命令：PowerShell 用 `Remove-Item -Recurse -Force .git`；Git Bash 用 `rm -rf .git`

### 步骤 2 · 重新初始化

```bat
git init -b main
```

预期输出：`Initialized empty Git repository in ...`

- 若报 `unknown switch 'b'`（git 版本过老）：先 `git init`，首次提交后再 `git branch -M main`

### 步骤 3 · 创建 .gitignore（关键！）

在项目根目录创建 `.gitignore`，至少覆盖三类内容：

```gitignore
# Python 缓存
__pycache__/
*.py[cod]

# IDE 本地配置
.idea/
.vscode/

# 模型权重 / 大文件（GitHub 单文件上限 100MB）
outputs/checkpoints/
*.pt
*.pth
*.ckpt
*.onnx
```

本项目完整版见根目录 [.gitignore](.gitignore)。

**验证排除是否生效**（必须有输出，显示命中的规则行号）：

```bat
git check-ignore -v outputs/checkpoints/best_cls.pt
```

### 步骤 4 · 干净提交

```bat
git add .
```

**提交前核对**（本步骤是最后防线）：

```bat
:: 暂存清单中不应出现 .pt / .pyc / .idea
git status --short

:: 双重确认：以下命令应无任何输出
git ls-files | findstr /i /c:".pt" /c:"pycache" /c:".idea"

:: 对象库体积应只有几十 MB
git count-objects -vH
```

确认无误后提交：

```bat
git commit -m "Initial commit: KUAKE-QIC 中文医疗意图分类 BERT vs LLM 对比项目"
```

- 身份已全局配置（`git config --global user.name / user.email`）则直接提交即可；
- 老版本 git 若默认分支为 master：`git branch -M main`

### 步骤 5 · 确认远程配置

```bat
git remote -v
```

三种情况：

```bat
:: 情况 A：重新 init 后 remote 为空 → 添加
git remote add origin https://github.com/<用户名>/<仓库名>.git

:: 情况 B：已有 origin 但地址要替换
git remote set-url origin https://github.com/<用户名>/<仓库名>.git

:: 情况 C：地址正确，无需操作
```

### 步骤 6 · 推送到远程

```bat
git push -u origin main
```

`-u` 绑定上游后，将来直接 `git push` 即可。

**首次推送的认证方式（二选一）：**

- 方式 A（推荐）· 弹窗授权：Git Credential Manager 窗口 → *Sign in with your browser* → 浏览器登录并授权
- 方式 B · PAT（个人访问令牌）：命令行要求输入时，Username 填 GitHub 用户名，Password 处**粘贴 PAT**（账号密码不可用）
  PAT 生成：GitHub 头像 → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token
  1. Token name：`git-push`；Expiration：按需
  2. Repository access：All repositories（或仅该仓库）
  3. Permissions → Repository permissions → **Contents: Read and write**（push 必需）
  4. Generate → 复制 `github_pat_...`（只显示一次）

### 步骤 7 · 验证

```bat
git log --oneline        :: 应只有一条 Initial commit
git status               :: working tree clean
git count-objects -vH    :: 体积应 < 100 MB
git ls-remote origin     :: 应出现 refs/heads/main
```

浏览器打开仓库页确认：文件齐全、**没有 checkpoints / *.pt**、仓库体积正常。

---

## 四、常见问题 FAQ

### Q1 · push 卡住 / 超时 / connection reset（网络问题）

配置代理（端口换成自己代理软件的实际端口，Clash 常见 7890、v2rayN 常见 10809）：

```bat
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

只让 GitHub 走代理（更精细）：

```bat
git config --global http.https://github.com.proxy http://127.0.0.1:7890
```

查看 / 删除代理：

```bat
git config --global --get-regexp proxy
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### Q2 · Authentication failed / 密码不正确

- GitHub 早已禁止账号密码推送，Password 必须填 PAT（见步骤 6 方式 B）
- 输错被缓存后先删凭据再重试：控制面板 → 用户账户 → 凭据管理器 → Windows 凭据 → 删除 `git:https://github.com`
- 或命令行查看：`cmdkey /list | findstr /i github`

### Q3 · `remote: error: File xxx is 390 MB; exceeds GitHub's file size limit of 100 MB`

大文件被 add 进了提交。补救（历史重写）：

```bat
:: 方案一：剔除后补一次提交（历史里仍残留大文件对象，不推荐）
git rm -r --cached outputs/checkpoints
git commit -m "Remove large checkpoints from index"

:: 方案二（推荐）：直接按本文档步骤 1~4 删除 .git 重建，一步到位
```

### Q4 · 建仓时勾选了 README 导致 push 被拒（non-fast-forward）

远程有一个自动生成的提交，与本地无共同历史。远程只有 README 时直接覆盖：

```bat
git push -u origin main --force
```

想保留双方提交：

```bat
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Q5 · 警告 `LF will be replaced by CRLF`

无害，可忽略；想根治：`git config --global core.autocrlf input`

### Q6 · 本地分支是 master 不是 main

```bat
git branch -M main
git push -u origin main
```

### Q7 · .gitignore 写了，文件还是被提交

`.gitignore` 只对未跟踪文件生效。已跟踪文件需先移出暂存区：

```bat
git rm -r --cached .idea
git rm --cached outputs/checkpoints/best_cls.pt
git commit -m "Apply .gitignore"
```

### Q8 · 只想换远程仓库地址、保留历史

无需清理时，一条命令即可切换远程：

```bat
git remote set-url origin https://github.com/<用户名>/<新仓库名>.git
git push -u origin main
```

### Q9 · 确实要上传大模型权重（.pt）怎么办

GitHub 不收 >100 MB 的普通文件，可选：

- **Git LFS**：`git lfs install` → `git lfs track "*.pt"` → `git add .gitattributes`；注意免费配额（存储 / 流量各 1 GB），本项目 1.17 GB 会超额
- **GitHub Releases** 附件（单文件 ≤ 2 GB）或 **Hugging Face Hub**（模型仓库更合适）

本项目建议保持排除：检查点属于可再生产物（跑训练脚本即可重新生成），4 MB 的 LoRA adapter 已保留上传用于展示。

---

## 五、完整命令速查（按顺序执行）

```bat
:: 0. 进入项目目录（按实际路径修改），先关闭 PyCharm
cd /d C:\Users\Latecomer\Desktop\共享\Project\kuaque-qic-bert-vs-llm

:: 1. 【体检】查看现状
git count-objects -vH
git ls-remote origin

:: 2. 【清理】删除旧历史（不可逆，确认后执行）
rmdir /s /q .git

:: 3. 【重建】初始化
git init -b main

:: 4. 【排除】确认 .gitignore 生效
git check-ignore -v outputs/checkpoints/best_cls.pt

:: 5. 【提交】暂存 + 核对 + 提交
git add .
git status --short
git ls-files | findstr /i /c:".pt" /c:"pycache" /c:".idea"
git commit -m "Initial commit: KUAKE-QIC 中文医疗意图分类 BERT vs LLM 对比项目"

:: 6. 【远程】配置仓库地址
git remote add origin https://github.com/Ohh-Sillage/kuaque-qic-bert-vs-llm.git
git remote -v

:: 7. 【推送】
git push -u origin main

:: 8. 【验证】
git log --oneline
git count-objects -vH
git ls-remote origin
```

---

## 六、本次实操记录

| 环节 | 执行前 | 执行后 |
|---|---|---|
| 提交历史 | 1 条 `first commit`（混入 1.17GB 大文件 + 12 个 .pyc） | 1 条干净的 `Initial commit` |
| `.git` 对象库 | **1.08 GiB**（97 objects） | **17.36 MiB**（82 objects；push 包 14.27 MiB） |
| 被跟踪大文件 | `checkpoints/*.pt` × 3、`__pycache__/*.pyc` × 12 | 无（由 `.gitignore` 排除） |
| 远程仓库 | origin 已配置，远程为空 | `refs/heads/main` → `fad813d`（新分支首次推送，无需 force） |
| 误报率风险 | push 必然被 GitHub 拒绝（>100MB 单文件） | 无（最大文件 ≈ 11 MB 的 tokenizer.json） |

> 推送验证：`git ls-remote origin` 与本地 `git log` 哈希一致（`fad813d`），`git status` 显示 working tree clean、与 `origin/main` 同步。

**遗留提醒**：`outputs/checkpoints/` 已在本地保留（CPU 训练 45 分钟才产出一个，删了要重跑）；它们只是不再进入 git，后续如需分享模型文件可走 Releases / 网盘。
