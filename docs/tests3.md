Para abordar cenários negativos, exceptions e simular casos de *Out of Memory* (OOM) nos testes unitários da classe `TorchDynamicShardInferenceEngine`, podemos expandir os testes existentes utilizando uma abordagem estruturada. A ideia é garantir que a classe seja robusta, lidando corretamente com erros e condições adversas que possam ocorrer em produção. A seguir, detalho como isso pode ser feito usando o framework `unittest` no Python, com exemplos práticos de código.

-----

## Abordagem Geral

1. **Cenários Negativos**: Testar situações em que entradas inválidas ou condições inesperadas ocorrem, como prompts ou tokens inválidos.
2. **Exceptions**: Simular falhas em dependências (ex.: `torch`, `ShardedGeneralModel`) e verificar se a classe trata essas exceções adequadamente.
3. **Casos de OOM**: Simular erros de memória insuficiente (ex.: `torch.cuda.OutOfMemoryError`) e validar se a classe libera recursos e registra os erros corretamente.

Vamos usar mocks (via `unittest.mock.patch`) para simular comportamentos de dependências externas e testar o comportamento da classe em cada caso.

-----

## 1. Testes para Cenários Negativos

Cenários negativos incluem situações como prompts ou tokens inválidos. Aqui estão exemplos de testes:

### Teste para `encode` com Prompt Inválido

Este teste verifica se a classe lida corretamente com uma falha ao codificar um prompt inválido:

```python
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from exo.inference.torch.sharded_inference_engine import TorchDynamicShardInferenceEngine, Shard

@patch('exo.inference.torch.sharded_inference_engine._resolve_tokenizer', return_value=MagicMock())
@patch('exo.inference.torch.sharded_inference_engine.ShardedGeneralModel')
async def test_encode_invalid_prompt(self, mock_model, mock_tokenizer):
    engine = TorchDynamicShardInferenceEngine(MagicMock())
    shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=2)
    mock_tokenizer.encode.side_effect = Exception("Invalid prompt")
    with self.assertRaises(Exception):
        await engine.encode(shard, "invalid_prompt")
```

### Teste para `decode` com Tokens Inválidos

Aqui, testamos a decodificação de tokens inválidos:

```python
@patch('exo.inference.torch.sharded_inference_engine._resolve_tokenizer', return_value=MagicMock())
@patch('exo.inference.torch.sharded_inference_engine.ShardedGeneralModel')
async def test_decode_invalid_tokens(self, mock_model, mock_tokenizer):
    engine = TorchDynamicShardInferenceEngine(MagicMock())
    shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=2)
    tokens = np.array([[999999]])  # Token inválido
    mock_tokenizer.decode.side_effect = Exception("Invalid tokens")
    with self.assertRaises(Exception):
        await engine.decode(shard, tokens)
```

-----

## 2. Testes para Exceptions

Vamos simular exceptions genéricas em métodos críticos como `infer_tensor` e `ensure_shard`.

### Teste para `infer_tensor` com `RuntimeError`

Este teste simula uma falha genérica no modelo durante a inferência:

```python
@patch('exo.inference.torch.sharded_inference_engine.ShardedGeneralModel')
async def test_infer_tensor_runtime_error(self, mock_model):
    engine = TorchDynamicShardInferenceEngine(MagicMock())
    shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=2)
    input_data = np.array([[1, 2, 3]])
    engine.model_instance = mock_model
    engine.inference_state = ShardInferenceState()
    engine.inference_state.tokens = torch.tensor([[1, 2, 3]]).to(engine.device)
    engine.inference_state.input_pos = torch.tensor([0]).to(engine.device)
    engine.inference_state.mask = torch.ones((1, 1, 1)).to(engine.device)
    mock_model.generate.side_effect = RuntimeError("Model failure")
    with self.assertRaises(RuntimeError):
        await engine.infer_tensor("test_request", shard, input_data)
    self.assertEqual(engine.out_of_memory_count, 0)  # Não deve ser contado como OOM
```

### Teste para `ensure_shard` com Falha ao Carregar Modelo

Aqui, simulamos uma falha ao carregar a configuração do modelo:

```python
@patch('exo.inference.torch.sharded_inference_engine.ShardDownloader')
@patch('exo.inference.torch.sharded_inference_engine.load_model_config', side_effect=Exception("Config load failure"))
async def test_ensure_shard_exception(self, mock_load_config, mock_shard_downloader):
    engine = TorchDynamicShardInferenceEngine(mock_shard_downloader)
    shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=2)
    with self.assertRaises(Exception):
        await engine.ensure_shard(shard)
    self.assertIsNone(engine.model_instance)
```

-----

## 3. Testes para Simular Casos de OOM

Para simular *Out of Memory* (OOM), usamos `torch.cuda.OutOfMemoryError` e verificamos se a classe limpa recursos e atualiza contadores adequadamente.

### Teste para `infer_tensor` com OOM

Este teste simula um erro de OOM e verifica o comportamento da classe:

```python
@patch('exo.inference.torch.sharded_inference_engine.ShardedGeneralModel')
@patch('exo.inference.torch.sharded_inference_engine.torch.cuda.empty_cache')
async def test_infer_tensor_oom(self, mock_empty_cache, mock_model):
    engine = TorchDynamicShardInferenceEngine(MagicMock())
    shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=2)
    input_data = np.array([[1, 2, 3]])
    engine.model_instance = mock_model
    engine.inference_state = ShardInferenceState()
    engine.inference_state.tokens = torch.tensor([[1, 2, 3]]).to(engine.device)
    engine.inference_state.input_pos = torch.tensor([0]).to(engine.device)
    engine.inference_state.mask = torch.ones((1, 1, 1)).to(engine.device)
    mock_model.generate.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
    result, state = await engine.infer_tensor("test_request", shard, input_data)
    self.assertIsNone(result)
    self.assertIsNone(state)
    self.assertEqual(engine.out_of_memory_count, 1)  # Incrementa contador de OOM
    mock_empty_cache.assert_called_once()  # Verifica se a memória foi limpa
    self.assertIsNone(engine.model_instance)  # Modelo deve ser liberado
    self.assertIsNone(engine.inference_state)  # Estado deve ser resetado
```

-----

## 4. Verificação de Logging

Além de tratar erros, é importante garantir que eles sejam registrados corretamente para facilitar a depuração.

### Teste para Logging em `infer_tensor` com OOM

```python
@patch('exo.inference.torch.sharded_inference_engine.ShardedGeneralModel')
async def test_infer_tensor_oom_logging(self, mock_model):
    engine = TorchDynamicShardInferenceEngine(MagicMock())
    shard = Shard(model_id="test_model", start_layer=0, end_layer=1, n_layers=2)
    input_data = np.array([[1, 2, 3]])
    engine.model_instance = mock_model
    engine.inference_state = ShardInferenceState()
    engine.inference_state.tokens = torch.tensor([[1, 2, 3]]).to(engine.device)
    engine.inference_state.input_pos = torch.tensor([0]).to(engine.device)
    engine.inference_state.mask = torch.ones((1, 1, 1)).to(engine.device)
    mock_model.generate.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")
    with self.assertLogs(level='ERROR') as log:
        await engine.infer_tensor("test_request", shard, input_data)
        self.assertIn("Out of memory on CUDA", log.output[0])  # Verifica mensagem de log
```

-----

## Conclusão

Com esses testes, cobrimos:

- **Cenários Negativos**: Entradas inválidas em `encode` e `decode`.
- **Exceptions**: Falhas genéricas em `infer_tensor` e `ensure_shard`.
- **Casos de OOM**: Simulação de `torch.cuda.OutOfMemoryError` com limpeza de recursos e logging.

Esses testes podem ser adicionados ao arquivo de testes existente e executados com o comando `python -m unittest`. Eles aumentam a robustez da classe `TorchDynamicShardInferenceEngine`, garantindo que ela lide adequadamente com erros e situações adversas em produção.
