
# QwenTalk

This is a project for QwenTalk.

## Model Acquisition

To use this project, you need to download the model first. You can use the following command to export the model:

```bash
optimum-cli export openvino --model Qwen/Qwen3-8B  --task text-generation-with-past --weight-format nf4 --sym --group-size -1 Qwen3-8B-nf4-ov --backup-precision int8_sym
```
