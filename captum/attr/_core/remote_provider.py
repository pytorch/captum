from abc import ABC, abstractmethod
from typing import Any, List, Optional
from captum._utils.typing import TokenizerLike
from openai import OpenAI
import os

class RemoteLLMProvider(ABC):
    """All remote LLM providers that offer logprob via API (like vLLM) extends this class."""
    
    api_url: str
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        **gen_args: Any
    ) -> str:
        """
        Args:
            prompt: The input prompt to generate from
            gen_args: Additional generation arguments
            
        Returns:
            The generated text.
        """
        ...
    
    @abstractmethod
    def get_logprobs(
        self, 
        input_prompt: str,
        target_str: str,
        tokenizer: Optional[TokenizerLike] = None
    ) -> List[float]:
        """
        Get the log probabilities for all tokens in the target string.
        
        Args:
            input_prompt: The input prompt
            target_str: The target string
            tokenizer: The tokenizer to use
            
        Returns:
            A list of log probabilities corresponding to each token in the target prompt.
            For a `target_str` of `t` tokens, this method returns a list of logprobs of length `k`.
        """
        ...

class VLLMProvider(RemoteLLMProvider):
    def __init__(self, api_url: str):
        assert api_url.strip() != "", "API URL is required"
        
        self.api_url = api_url
        self.client = OpenAI(base_url=self.api_url,
                             api_key=os.getenv("OPENAI_API_KEY", "EMPTY")
                            )
        self.model_name = self.client.models.list().data[0].id
        

    def generate(self, prompt: str, **gen_args: Any) -> str:
        if not 'max_tokens' in gen_args:
            gen_args['max_tokens'] = gen_args.pop('max_new_tokens', 25)
        if 'do_sample' in gen_args:
            gen_args.pop('do_sample')

        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            **gen_args
        )
        
        return response.choices[0].text
    
    def get_logprobs(self, input_prompt: str, target_str: str, tokenizer: Optional[TokenizerLike] = None) -> List[float]:
        assert tokenizer is not None, "Tokenizer is required for VLLM provider"
        
        num_target_str_tokens = len(tokenizer.encode(target_str, add_special_tokens=False))
        
        prompt = input_prompt + target_str
        
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            temperature=0.0,
            max_tokens=1,
            extra_body={"prompt_logprobs": 0}
        )
        prompt_logprobs = []
        for probs in response.choices[0].prompt_logprobs[1:]:
            prompt_logprobs.append(list(probs.values())[0]['logprob'])
        
        return prompt_logprobs[-num_target_str_tokens:]
        