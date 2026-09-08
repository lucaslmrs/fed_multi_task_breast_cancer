# Aceleração de treinamento: BF16 e runtime concorrente

## O que mudou

Novas runs usam `training.precision: bf16`. O autocast reduz a precisão apenas de operações
elegíveis de forward, loss e inferência; parâmetros do modelo e estado do otimizador continuam em
FP32. Configurações antigas que não possuem `training.precision` continuam resolvendo como FP32.

BF16 é intencionalmente estrito: uma run BF16 falha antes de treinar se CUDA estiver ausente ou se
a GPU não tiver suporte BF16 nativo. Smokes de CPU sobrescrevem explicitamente para FP32.

```yaml
training:
  precision: bf16
  cuda_benchmark: true

runtime:
  inference_batch_size: 32
  dataloader:
    num_workers: 4
    federated_num_workers: 0
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 2
  federated:
    ray_num_cpus: 8
    client_resources: {num_cpus: 2, num_gpus: 0.5}
  telemetry:
    enabled: true
    gpu_interval_seconds: 1.0
```

Dois atores Ray podem compartilhar uma GPU porque cada cliente reserva `0.5` GPU. Os loaders
federados começam com zero workers: os próprios atores sobrepõem CPU/I/O e GPU sem criar uma árvore
adicional de processos. Os caminhos clássicos usam quatro workers, memória pinada, workers
persistentes e prefetch 2. Cópias host→GPU usam `non_blocking=True`.

## Comparabilidade científica

`training.precision` e `training.cuda_benchmark` pertencem ao hash científico e impedem resume
cruzado entre FP32 e BF16. O bloco `runtime` e a telemetria são operacionais e ficam fora dos hashes
de comparabilidade/resume. Batch de treino, partições, sementes, critérios, orçamento e validação
por época/rodada continuam científicos.

- `studies/example_multi_dataset.yaml`: manifesto padrão, protocolo BF16 com
  `cuda_benchmark: true` fixado em `config_overrides`.
- Para um protocolo FP32, copie o manifesto e troque `training.precision` para `fp32` (e
  `cuda_benchmark` para `false`); precisão numérica não cria nem altera partições, mas entra no hash
  científico, então os dois estudos não compartilham resume.

No federado, a validação continua ocorrendo depois de cada agregação. A avaliação pós-treino local e
o antigo `best.pt` diagnóstico foram removidos; o artefato final continua sendo o tronco global da
última rodada combinado com `state.pt` personalizado e estado do otimizador por cliente. No treino
clássico de segmentação, o teste ocorre uma única vez, depois de carregar o melhor checkpoint.

## Artefatos de desempenho

Cada run grava dados de forma bufferizada, sem escrita por batch:

- `gpu_telemetry.csv`: utilização de GPU/memória, VRAM usada/total, potência e temperatura a cada
  segundo;
- `runtime_events.csv`: duração, batches, exemplos/s e picos de memória CUDA por fase;
- `fold_*/client_*/training_history.csv`: duração, exemplos/s e picos por cliente/rodada, junto às
  perdas e métricas já auditadas.

Se NVML não puder inicializar, a telemetria é desativada com aviso e a run científica continua.
Uma perda NaN/Inf, porém, interrompe o treino imediatamente com contexto da fase.

## Sequência copiável

```bash
cd /home/lucas/fed_multi_task_breast_cancer
source .venv/bin/activate

# Verifica configuração e todos os braços sem treinar nem tocar nas partições.
python -m src.experiments.study_runner --dry-run

# Smokes usam FP32/CPU deliberadamente.
python -m src.experiments.study_runner --smoke --seed-profile operational
python -m scripts.smoke_federated --setup both --samples 2

# Benchmark de aceite na GPU: FP32 sequencial versus BF16/2 clientes.
# Usa 5 rodadas para amortizar startup, o mesmo master congelado e remove os
# checkpoints temporários.
python -m scripts.benchmark_training_runtime

# Estudo de exemplo (BF16).
python -m src.experiments.study_runner --seed-profile operational

# Qualquer outro manifesto em studies/.
python -m src.experiments.study_runner \
  --manifest studies/<meu_estudo>.yaml \
  --seed-profile operational
```

O benchmark falha se o ganho ponta a ponta for menor que 25%, se houver NaN/Inf ou se a diferença
absoluta média nas métricas primárias pareadas (Dice para segmentação e balanced accuracy para
classificação) ultrapassar 0,02. O relatório compacto fica em
`runs/benchmarks/<timestamp>_paired_fp32_bf16/`; use `--keep-runs` apenas se precisar inspecionar os
checkpoints grandes.

## Diagnóstico rápido

- GPU baixa e CPU saturada: confira `examples_per_second`, workers e frequência de lotes em
  `runtime_events.csv`.
- GPU baixa com pausas regulares: compare as fases Flower e pós-agregação; a validação em toda rodada
  é preservada por desenho.
- VRAM perto do limite: volte temporariamente a `num_gpus: 1.0`; não reduza o batch científico para
  comparar a mesma run.
- Falha BF16: rode `python -c "import torch; print(torch.cuda.is_bf16_supported())"`. CPU deve usar
  `training.precision: fp32`.

## Limite de memória do WSL

No Windows, crie manualmente `C:\Users\rluca\.wslconfig` com:

```ini
[wsl2]
memory=24GB
```

Depois, em PowerShell, aplique a mudança com:

```powershell
wsl --shutdown
```

Esse comando encerra as distribuições WSL em execução. Ele é uma etapa manual e nunca é executado
automaticamente pelos scripts deste repositório.
