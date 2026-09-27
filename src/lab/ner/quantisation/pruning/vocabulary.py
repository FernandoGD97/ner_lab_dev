"""Conservative tail-only WordPiece pruning and multilingual analysis."""
from pathlib import Path
from ..artifacts import prepare_output,write_artifact_metadata
from ..base import CompressionMethod
from ..metadata import *
from ..model import checkpoint_size, load_checkpoint
VOCAB_META=MethodMetadata('vocabulary-pruning','vocabulary',CompressionCategory.MODEL_COMPRESSION,TrainingRequirement.NONE,
 fidelity=Fidelity.GENERIC_EQUIVALENT,limitations=('Phase 1 mutation is tail-only WordPiece; SentencePiece/XLM-R is analysis-only to avoid changing IDs.'),
 publications=(Publication('vocab-pruning','Vocabulary Reduction for Neural Machine Translation','https://aclanthology.org/P17-2086/'),Publication('rembert','REMBERT','https://arxiv.org/abs/2010.12821')))
def analyse_vocabulary(source,texts):
 _,tok,_=load_checkpoint(source); before=[len(tok(t,add_special_tokens=False)['input_ids']) for t in texts]
 return {'original_vocabulary_size':len(tok),'tokens_per_document':sum(before)/len(before) if before else 0.0,'subwords_per_word':sum(before)/sum(max(1,len(t.split())) for t in texts) if texts else 0.0,'operation':'pruning_existing_vocabulary_not_training_a_new_tokenizer'}
class VocabularyPruningMethod(CompressionMethod):
 metadata=VOCAB_META
 def apply(self,source,output,keep_tokens=(),task='auto',**kwargs):
  model,tok,_=load_checkpoint(source,task)
  vocab=tok.get_vocab(); required=set(tok.all_special_tokens)|set(keep_tokens)
  missing=required-set(vocab)
  if missing: raise ValueError(f'Tokens are absent from source vocabulary: {sorted(missing)}')
  if not type(tok).__name__.startswith('BertTokenizer'): raise ValueError('Safe Phase 1 vocabulary mutation supports WordPiece/BertTokenizer only; XLM-R/SentencePiece is analysis-only.')
  new_size=max(vocab[t] for t in required)+1
  if new_size>=len(vocab): raise ValueError('Selected tokens do not permit tail pruning while preserving every token ID.')
  out=prepare_output(source,output); old_size=len(vocab); old_bytes=checkpoint_size(source)
  model.resize_token_embeddings(new_size); model.config.vocab_size=new_size; model.save_pretrained(out,safe_serialization=True)
  tokens=[None]*old_size
  for token,index in vocab.items(): tokens[index]=token
  (out/'vocab.txt').write_text('\n'.join(tokens[:new_size])+'\n',encoding='utf8'); tok.save_pretrained(out)
  # A fast tokenizer JSON embeds the old vocabulary; force reconstruction from
  # the exact WordPiece prefix instead of leaving inconsistent dual sources.
  (out/'tokenizer.json').unlink(missing_ok=True)
  (out/'vocab.txt').write_text('\n'.join(tokens[:new_size])+'\n',encoding='utf8')
  details={'original_vocabulary_size':old_size,'new_vocabulary_size':new_size,'embedding_parameters_removed':(old_size-new_size)*model.get_input_embeddings().embedding_dim,'serialized_bytes_saved':old_bytes-checkpoint_size(out),'token_ids_preserved':True}
  write_artifact_metadata(source,out,'vocabulary-pruning',self.metadata,details=details); self.validate(out); return out
