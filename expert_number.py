import os
import torch
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import tqdm
import json
import pickle


def exponential_scaling(values, target_sum, exponent):
    values = np.array(values)
    scaled_values = np.power(values, exponent)
    scaled_integers = np.round((scaled_values / scaled_values.sum()) * target_sum).astype(int)

    while scaled_integers.sum() != target_sum:
        difference = target_sum - scaled_integers.sum()
        if difference > 0:
            scaled_integers[np.argmin(scaled_values - scaled_integers)] += 1
        else:
            scaled_integers[np.argmax(scaled_values - scaled_integers)] -= 1

    return scaled_integers


def fix_finger(w, bins=100, pl_fitting=True, EVALS_THRESH=1e-4, filter_zeros=False):
    eigs = torch.square(torch.linalg.svdvals(w).flatten())
    eigs, _ = torch.sort(eigs, descending=False)

    if filter_zeros:
        nz_eigs = eigs[eigs > EVALS_THRESH]
        N = len(nz_eigs)
    else:
        # print(f"{name} Skip Filter Zero")
        nz_eigs = eigs
        N = len(nz_eigs)

    # Handle very small eigenvalues
    min_eig = nz_eigs.min()
    if min_eig <= 0:
        # If we have non-positive eigenvalues, return default value
        return torch.tensor(1.0, device=eigs.device)
    
    log_nz_eigs = torch.log(nz_eigs)
    alphas = torch.zeros(N - 1)
    Ds = torch.ones(N - 1)
    if pl_fitting:
        # Clip eigenvalues to avoid -inf in log10
        safe_eigs = torch.clamp(nz_eigs, min=1e-100)
        hist_nz_eigs = torch.log10(safe_eigs)
        min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
        
        # Skip histogram if range is invalid
        if torch.isfinite(min_e) and torch.isfinite(max_e) and min_e < max_e:
            counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
            boundaries = torch.linspace(min_e, max_e, bins + 1)
            h = counts, boundaries
            ih = torch.argmax(h[0])
            xmin2 = 10 ** h[1][ih]
            xmin_min = torch.log10(0.95 * xmin2)
            xmin_max = 1.5 * xmin2
        else:
            # If histogram range is invalid, skip pl_fitting
            pl_fitting = False
            xmin_min = None
            xmin_max = None

    for i, xmin in enumerate(nz_eigs[:-1]):
        if pl_fitting == True:
            if xmin < xmin_min:
                continue
            if xmin > xmin_max:
                break

        n = float(N - i)
        #seq = torch.arange(n).cuda(nz_eigs.device)
        alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
        alphas[i] = alpha
        if alpha > 1:
            seq = torch.arange(n, device=nz_eigs.device)
            Ds[i] = torch.max(torch.abs(
                1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n
            ))

    if len(Ds) > 0:
        min_D_index = torch.argmin(Ds)
        final_alpha = alphas[min_D_index]
        return final_alpha
    else:
        # Return a default value as tensor when no valid alpha is found
        return torch.tensor(1.0, device=eigs.device)  # Default alpha value as tensor

class WrappedGPT:
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.data.shape
        self.scaler_row = torch.zeros(self.columns, device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, torch.nn.Linear) and len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler_row += torch.norm(inp.float(), p=2, dim=1) ** 2 / self.nsamples


def find_layers(module, layers=[nn.Linear], name=''):

    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def save_checkpoint(checkpoint_path, all_layer_alpha, current_layer):
    checkpoint = {
        'all_layer_alpha': all_layer_alpha,
        'current_layer': current_layer
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint['all_layer_alpha'], checkpoint['current_layer']

def calculate_expert(model, model_path, save_path=None, start_at=0, end_at=None):
    import time
    start_time = time.time()
    
    # Initialize or load checkpoint
    if save_path and os.path.exists(save_path):
        all_layer_alpha, start_at = load_checkpoint(save_path)
        print(f"Resuming from checkpoint at layer {start_at}")
    else:
        all_layer_alpha = []
    
    layers = model.model.layers
    end_at = end_at if end_at is not None else len(layers)

    # Handle meta device parameters
    if any(p.is_meta for p in model.parameters()):
        # If using device_map, we need to load parameters as we go
        print("Model contains meta device parameters - using streaming approach")
    
    # Add progress bar
    from tqdm import tqdm
    pbar = tqdm(total=end_at, initial=start_at, desc="Processing layers")
    
    for i in range(start_at, end_at):
        layer = layers[i]
        try:
            # Move only the current layer to GPU, handling meta device
            # First load any meta parameters
            if any(p.is_meta for p in layer.parameters()):
                # Create a new state dict with initialized parameters
                new_state_dict = {}
                for name, param in layer.state_dict().items():
                    if param.is_meta:
                        # Initialize parameter with proper shape and dtype
                        param = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
                    new_state_dict[name] = param
                # Load the new state dict with assign=True
                layer.load_state_dict(new_state_dict, strict=False, assign=True)
            
            # Now move to GPU
            layer = layer.cuda()
            
            subset = find_layers(layer)
            pbar.set_postfix_str(f"Layer {i+1} - {len(subset)} linear layers")
            
            if subset:  # Only process if subset is not empty
                layer_final_alpha = []
                for name in subset:
                    try:
                        # Process each linear layer individually, handling meta device
                        linear_layer = subset[name]
                        # Handle meta parameters for linear layer
                        if any(p.is_meta for p in linear_layer.parameters()):
                            # Create a new state dict with initialized parameters
                            new_state_dict = {}
                            for name, param in linear_layer.state_dict().items():
                                if param.is_meta:
                                    # Initialize parameter with proper shape and dtype
                                    param = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
                                new_state_dict[name] = param
                            # Load the new state dict with assign=True
                            linear_layer.load_state_dict(new_state_dict, strict=False, assign=True)
                        
                        # Now move to GPU
                        linear_layer = linear_layer.cuda()
                        alpha = fix_finger(linear_layer.weight.data.float())
                        # Ensure alpha is on GPU before appending
                        layer_final_alpha.append(alpha.cuda())
                        # Move layer back to CPU
                        linear_layer.cpu()
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print(f"\nWarning: CUDA OOM processing {name}, using default alpha")
                            layer_final_alpha.append(torch.tensor(1.0, device='cuda').cuda())
                        else:
                            raise
                            
                if layer_final_alpha:  # Check if we got any alpha values
                    mean_alpha = torch.stack(layer_final_alpha).mean().item()
                    all_layer_alpha.append(mean_alpha)
                    print(f"PL_Alpha_Hill for layer {i+1}: {mean_alpha:.4f}")
                else:
                    all_layer_alpha.append(1.0)  # Default value if no alpha could be calculated
                    print(f"PL_Alpha_Hill for layer {i+1}: 1.0 (default)")
            else:
                all_layer_alpha.append(1.0)  # Default value for empty layers
                # AI Add warning that default PL_Alpha_Hill for layer {i+1}: ==1 AI!
            
            # Move current layer back to CPU
            layer.cpu()
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print(f"\nWarning: CUDA OOM processing layer {i+1}, using default alpha")
                all_layer_alpha.append(1.0)
            else:
                raise
                
        pbar.update(1)
            
        # Save checkpoint after each layer
        if save_path:
            save_checkpoint(save_path, all_layer_alpha, i+1)
        
    pbar.close()
    
    # Print timing information
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    
    return all_layer_alpha





def get_llm(model_name):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_buffers=True
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="mistralai/Mistral-7B-v0.1", type=str)
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--beta', type=float, default=2.5)
    parser.add_argument('--target_sum', type=int, default=160)
    parser.add_argument('--save', type=str, help='Path to save checkpoint file')
    parser.add_argument('--load', type=str, help='Path to load checkpoint file')
    parser.add_argument('--start-at', type=int, default=0, help='Layer index to start processing from')
    parser.add_argument('--end-at', type=int, help='Layer index to stop processing at')


    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model = get_llm(args.model)
    model.eval()

    # Determine checkpoint path
    checkpoint_path = args.load if args.load else args.save
    
    distribution = calculate_expert(
        model, 
        args.model,
        save_path=checkpoint_path,
        start_at=args.start_at,
        end_at=args.end_at
    )

    print("Distribution:", distribution)
    quantized_vector = exponential_scaling(distribution, args.target_sum, args.beta)
    print("Total expert number:", sum(quantized_vector))
    print("expert number: ", ','.join(map(str, quantized_vector)))

    topkk = [2 if n > 1 else 1 for n in quantized_vector]
    topkk = ','.join(map(str, topkk))
    print("top_k: ", topkk)

if __name__ == '__main__':
    main()
