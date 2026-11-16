# 数据来源、格式与精确划分

本仓库没有图片、标注正文、gold answers、预测或数据压缩包。`tessfold/sources/`
只保存公开文件下载地址的组成信息、固定 revision、内容校验值及行索引配方。
原始数据的许可和图像来源条款由各数据集提供方决定。

## 自己准备清单

`train_rows` / `dev_rows` / `eval_rows` 各指向一个外部 JSONL。每行格式为：

```json
{"id":"unique-question-id","document_id":"image-content-id","image":"/absolute/path/to/page.png","question":"Your question","answers":["Gold answer"],"task":"DocVQA"}
```

这只是格式示意，不作为样本数据分发。支持 `DocVQA`、`InfoVQA`、`ChartQA`、
`OCRBench`、`DeepForm`、`KLC`、`WTQ`、`DT-VQA`。
相对图片路径从配置的 `data_root` 解析；配置内其他相对路径从运行目录解析。

三种 split 的题目 ID、document_id、图片路径和实际图片内容都不能交叉。
同一个 document_id 可有多个问题，但必须对应同一图片。`validate` 会检查这些条件，
训练时也会重新检查并记录 SHA-256。测试集从不参与参数或 epoch 选择。

## 原公开来源

| 任务 | 来源 |
|---|---|
| DocVQA、InfoVQA | `lmms-lab/DocVQA` 的 DocVQA / InfographicVQA |
| ChartQA | `ahmed-masry/ChartQA` |
| OCRBench | `echo840/OCRBench`，仅正式评测 |
| DeepForm、KLC、WTQ | `echo840/Monkey_Data` 原 DUE_Benchmark 图像与问答 |
| DT-VQA | `shuozhang2/DT-VQA` 原图片及官方问答 |

固定来源 revision 和逐文件 hash 见 `datasets.files.json`、
`monkey_annotation.source.json`、`dtvqa.source.json`。
DT-VQA 以 Hugging Face 模型仓库形式托管，下载 URL 因此不带 `/datasets/`。

## 从公开来源重建

下载是显式操作，需要较多时间和磁盘；不会在安装或测试时触发。
原 Monkey 的 multipart ZIP 使用 HTTP Range 按需读取，服务端必须支持精确 Range。

```bash
.venv/bin/python -m pip install -r requirements-legacy.txt
.venv/bin/python -m tessfold download-data
# 如果网络需要镜像：
# .venv/bin/python -m tessfold download-data --endpoint https://hf-mirror.com
.venv/bin/python -m scripts.prepare_full_vqa
.venv/bin/python -m scripts.append_due_vqa
.venv/bin/python -m scripts.append_dtvqa
```

原预处理还运行 `scripts.preflight_full_data`：过滤空监督、核查图像内容与解码像素重复，
并记录序列长度。该历史脚本需要 `configs/full.json` 中指定的 Qwen3-1.7B tokenizer
位于本地 `models/Qwen3-1.7B`。可用 Hugging Face API 下载 tokenizer 文件：

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-1.7B', local_dir='models/Qwen3-1.7B',
                  allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'])
PY
.venv/bin/python -m scripts.preflight_full_data
.venv/bin/python -m tessfold prepare-cohorts
```

`prepare-cohorts` 先核对转换后的 `data/full/{train,val,test}.jsonl` 的原固定 SHA-256，
再按行索引生成 `data/train.jsonl`（20,000 QA）、`data/dev.jsonl`（56 QA）和
`data/eval.jsonl`（23,565 QA），并逐一核对生成文件的固定 SHA-256。
这三组输出由 `.gitignore` 忽略。它不会为了通过校验重新抽样或悄悄替换 cohort。

若已经在另一个目录完成相同公开转换，可直接：

```bash
.venv/bin/python -m tessfold prepare-cohorts --source-root /path/to/converted/project
```

然后把配置 `data_root` 指向该源目录，或自行映射清单内的图片路径。
如果来源版本、转换顺序、Pillow 解码结果或预处理过滤发生变化，源 hash 可能不匹配；
这时应显式定义新 split 并报告协议变化，不能宣称重建了原固定划分。

原 tokenizer 下载示例未固定 revision，只用于预处理长度统计；该 tokenizer 版本必须
另行记录。固定 cohort 的来源和最终输出 SHA-256 才是本重建入口的强制一致性边界。
