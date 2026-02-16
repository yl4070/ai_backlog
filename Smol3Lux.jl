### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 5c0a8852-087d-11f1-950b-cdf741336217
# ╠═╡ show_logs = false
begin
	using Lux, SafeTensors
	using HuggingFaceTokenizers, JSON3
end

# ╔═╡ b7fdbfd6-a3a7-402d-bb27-f4f8258ff7fa
using Reactant

# ╔═╡ 63b217ae-3900-47ee-be17-bcabc038858a
using BFloat16s, Random

# ╔═╡ db7b31c7-7cea-443a-9107-b0f0fe44967a
using PythonCall

# ╔═╡ eb64305e-281b-4cf4-9944-22aed864f66e
using ConcreteStructs

# ╔═╡ 804f121f-6c51-4e41-b366-c4410cc05ac8
using ProgressLogging

# ╔═╡ 18606732-3cbd-4e15-8ff3-323181368342
begin
function get_padded_size(seq_len::Int, context_length::Int)
    return min(max(512, nextpow(2, seq_len)), context_length)
end

function padded_input_and_mask_len(x::AbstractMatrix, v, cfg, pad_token_id)
    return padded_input_and_mask_len(
        x, v, get_padded_size(size(x, 1) + v !== nothing, cfg.max_position_embeddings), pad_token_id
    )
end

function padded_input_and_mask_len(x::AbstractMatrix, v, padded_sz::Int, pad_token_id)
    if padded_sz > size(x, 1)
        x_padded = similar(x, (padded_sz, size(x, 2)))
        x_padded[1:size(x, 1), :] .= x
        if v === nothing
            x_padded[(size(x, 1) + 1):end, :] .= pad_token_id
        else
            x_padded[(size(x, 1) + 1), :] = v[1, :]
            x_padded[(size(x, 1) + 2):end, :] .= pad_token_id
        end
    else
        x_padded = x
    end
    return (
        x_padded,
        Reactant.promote_to(
            Reactant.TracedRNumber{Int32}, padded_sz - (size(x, 1) + (v !== nothing))
        ),
    )
end
end

# ╔═╡ 0f6feaa5-3d27-4a40-888d-e49f2d071e9c
Reactant.set_default_backend("gpu")

# ╔═╡ a7c4f7ac-701b-4f05-8132-3dfec70585c7
begin
struct SmolTokenizer
    tokenizer::Tokenizer
    special_to_id::Dict{String,Int32}
    pad_token_id::Int32
    eos_token_id::Int32
    apply_chat_template::Bool
    add_generation_prompt::Bool
    add_thinking::Bool
end

const SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>"
]

const SPLIT_RE = r"(<\|[^>]+?\|>)"

token_to_id(tokenizer::SmolTokenizer, s) = token_to_id(tokenizer.tokenizer, s)
function token_to_id(tokenizer::Tokenizer, s)
    return pyconvert(Int32, tokenizer.py_tokenizer.token_to_id(s)) + Int32(1)
end

function split_with_delims(text::String, re::Regex)
    parts = String[]
    last_end = 1
    for m in eachmatch(re, text)
        if m.offset > last_end
            push!(parts, text[last_end:(m.offset - 1)])
        elseif m.offset == 1
            push!(parts, "")
        end
        push!(parts, m.match)
        last_end = m.offset + length(m.match)
    end
    if last_end ≤ lastindex(text)
        push!(parts, text[last_end:end])
    end
    return parts
end

function SmolTokenizer(
    tokenizer_file_path::String;
    repo_id=nothing,
    apply_chat_template::Bool=true,
    add_generation_prompt::Bool=false,
    add_thinking::Bool=false,
)
    tok = HuggingFaceTokenizers.from_file(Tokenizer, tokenizer_file_path)
    special_to_id = Dict(s => token_to_id(tok, s) for s in SPECIALS)
    pad_token_id = special_to_id["<|endoftext|>"]
    eos_token_id = pad_token_id
    if repo_id !== nothing && !occursin("Base", repo_id)
        eos_token = "<|im_end|>"
    else
        eos_token = "<|endoftext|>"
    end
    if haskey(special_to_id, eos_token)
        eos_token_id = special_to_id[eos_token]
    end
    return SmolTokenizer(
        tok,
        special_to_id,
        pad_token_id,
        eos_token_id,
        apply_chat_template,
        add_generation_prompt,
        add_thinking,
    )
end

function wrap_chat(tokenizer::SmolTokenizer, user_msg::AbstractString)
    s = "<|im_start|>user\n$(user_msg)<|im_end|>\n"
    if tokenizer.add_generation_prompt
        s *= "<|im_start|>assistant"
        if tokenizer.add_thinking
            s *= "\n"
        else
            s *= "\n<think>\n\n</think>\n\n"
        end
    end
    return s
end

function HuggingFaceTokenizers.encode(
    tok::SmolTokenizer, text; chat_wrapped::Bool=tok.apply_chat_template
)
    stripped = strip(text)
    if haskey(tok.special_to_id, stripped) && !occursin('\n', stripped)
        return [tok.special_to_id[stripped]]
    end

    chat_wrapped && (text = wrap_chat(tok, text))

    ids = Int32[]
    for part in filter(!isempty, split_with_delims(text, SPLIT_RE))
        if haskey(tok.special_to_id, part)
            push!(ids, tok.special_to_id[part])
        else
            append!(ids, encode(tok.tokenizer, string(part)).ids .+ Int16(1))
        end
    end
    return ids
end

function HuggingFaceTokenizers.decode(tok::SmolTokenizer, ids::Vector{<:Integer})
    return decode(tok.tokenizer, ids .- Int16(1); skip_special_tokens=false)
end

end

# ╔═╡ 27f087ae-876b-4a7d-920d-1d507f47d294


# ╔═╡ d8a6c364-ef26-4b70-ab68-72c46ae2c7ec
weights_path = joinpath(@__DIR__, "weights") 

# ╔═╡ d10633a6-44dd-4a9a-9881-f36a15a8e57d
smtok = SmolTokenizer(joinpath(weights_path, "tokenizer.json"))

# ╔═╡ 9999bb0f-2d52-4777-9334-6c2e58141721
weights_dict = load_safetensors(joinpath(weights_path, "model.safetensors"))

# ╔═╡ 9bcb4599-e27b-48c1-a2f0-78d4e1972888
begin
altRMSNorm(emb_dim::Int) = AlternatePrecision{Float32}(RMSNorm(emb_dim; affine = false))

function apply_rope(x::AbstractArray{T}, cos_cache, sin_cache) where {T}
    return T.(apply_rotary_embedding(x, cos_cache, sin_cache; seq_dim=3))
end

function GroupedQueryAttention(cfg)

	d_in = cfg.hidden_size
	num_heads = cfg.num_attention_heads
	num_kv_groups = cfg.num_key_value_heads

	max_pos = cfg.max_position_embeddings
    θ = cfg.rope_theta
	
    @assert num_heads % num_kv_groups == 0 
	@assert d_in % num_heads == 0 "`d_in` must be divisible by `num_heads`"
                                       
    head_dim = d_in ÷ num_heads
    d_out = num_heads * head_dim


	@compact(;
		q_proj = Dense(d_in, d_out; use_bias=false),
        k_proj = Dense(d_in, num_kv_groups * head_dim; use_bias=false),
        v_proj = Dense(d_in, num_kv_groups * head_dim; use_bias=false),
        o_proj = Dense(d_out, d_in; use_bias=false),
        q_norm = altRMSNorm(head_dim),
        k_norm = altRMSNorm(head_dim),
		cache = @init_fn(rng -> compute_rotary_embedding_params(
        head_dim, max_pos; base=θ, dtype=Float32
    ), :state)
	) do x

		_, num_tokens, batch = size(x)
		
		queries = q_proj(x) 
		keys = k_proj(x)
		values = v_proj(x)

		#reshape to 4 dims
		queries = reshape(queries, head_dim, num_heads, num_tokens, batch)
    	keys = reshape(keys, head_dim, num_kv_groups, num_tokens, batch)
    	values = reshape(values, head_dim, num_kv_groups, num_tokens, batch)

		queries = q_norm(queries)
		keys = k_norm(keys)

		
		queries = apply_rope(queries, cache[:cos_cache], cache[:sin_cache])
		keys = apply_rope(keys, cache[:cos_cache], cache[:sin_cache])

		context = scaled_dot_product_attention(
			queries, keys, values;
		)[1]

		context = reshape(context,head_dim * num_heads, num_tokens, batch)

		proj = o_proj(context)
		
		@return proj
	end
end

end

# ╔═╡ 2e05fefb-ed47-4e51-b704-bc744c3a7fbe
cfg = JSON3.read(joinpath(weights_path, "config.json"))

# ╔═╡ 1a0285af-a049-4c79-955b-a8c4ff7e3f90
begin
function predict_next_token(
    model, token_ids::AbstractMatrix{T}, input_mask_len, ps, st
) where {T}
    logits, stₙ = model(token_ids, ps, st)
    predictions = T.(argmax(logits[:, end - input_mask_len, :]; dims=1))
    predictions = mod1.(predictions, T(size(logits, 1)))
    return predictions, stₙ
end

function update_token_ids_and_mask!(
    padded_token_ids, input_mask_len, cur_num_tokens, next_token
)
    next_token_idx = safe_increment(cur_num_tokens)
    padded_token_ids[next_token_idx, :] = next_token[1, :]
    return input_mask_len - eltype(input_mask_len)(1), next_token_idx
end

function update_token_ids_with_shift!(token_ids, next_token)
    token_ids[1:(end - 1), :] = token_ids[2:end, :]
    token_ids[end, :] = next_token[1, :]
    return nothing
end

safe_increment(x) = x + one(x)

mutable struct CachedReactantThunks
    cache::Dict{AbstractString,Dict{Int,NTuple{3,Reactant.Compiler.Thunk}}}
    increment_fn::Union{Nothing,Reactant.Compiler.Thunk}
end

function CachedReactantThunks()
    return CachedReactantThunks(
        Dict{AbstractString,Dict{Int,NTuple{3,Reactant.Compiler.Thunk}}}(), nothing
    )
end

function cache_and_retrieve!(
	label::AbstractString,
    cache::CachedReactantThunks,
    len::Integer,
    model,
    padded_token_ids,
    input_mask_len,
    ps,
    st,
    next_token,
    cur_num_tokens_traced,
)


    predict_next_token_compiled = @compile predict_next_token(
        model, padded_token_ids, input_mask_len, ps, st
    )
    update_fn1! = @compile update_token_ids_and_mask!(
        padded_token_ids, input_mask_len, cur_num_tokens_traced, next_token
    )
    update_fn2! = @compile update_token_ids_with_shift!(padded_token_ids, next_token)

    if !haskey(cache.cache, label)
        cache.cache[label] = Dict{Int,NTuple{3,Reactant.Compiler.Thunk}}()
    end

    return cache.cache[label][len] = (
        predict_next_token_compiled, update_fn1!, update_fn2!
    )
end

const CACHED_THUNKS = CachedReactantThunks()

generate_text(args...; kwargs...) = generate_text!(CACHED_THUNKS, args...; kwargs...)

function generate_text!(
    compile_cache::CachedReactantThunks,
    model,
    prompt::String,
    ps,
    st,
    max_new_tokens,
    tokenizer,
)
    token_ids = Reactant.to_rarray(reshape(encode(tokenizer, prompt), :, 1))

    cur_num_tokens = size(token_ids, 1)
    max_context_length = cfg.max_position_embeddings
    cur_compiled_fn_token_len = get_padded_size(cur_num_tokens, max_context_length)

    padded_token_ids, input_mask_len = @jit padded_input_and_mask_len(
        token_ids, nothing, cur_compiled_fn_token_len, tokenizer.pad_token_id
    )
    cur_num_tokens_traced = ConcreteRNumber{Int32}(cur_num_tokens)

    next_token = get_device(ps)(rand(Int32, 1, size(padded_token_ids, 2)))

    (predict_next_token_compiled, update_fn1!, update_fn2!) = cache_and_retrieve!(
		"model",
        compile_cache,
        cur_compiled_fn_token_len,
        model,
        padded_token_ids,
        input_mask_len,
        ps,
        st,
        next_token,
        cur_num_tokens_traced,
    )

    if compile_cache.increment_fn === nothing
        compile_cache.increment_fn = @compile safe_increment(cur_num_tokens_traced)
    end

    start_time = time()
    compile_time = 0.0
    ntokens_generated = 0

    for _ in 1:max_new_tokens
        new_compiled_fn_token_len = get_padded_size(cur_num_tokens, max_context_length)
        if new_compiled_fn_token_len != cur_compiled_fn_token_len
            compile_start_time = time()
            cur_compiled_fn_token_len = new_compiled_fn_token_len
            padded_token_ids, input_mask_len = @jit padded_input_and_mask_len(
                padded_token_ids,
                next_token,
                cur_compiled_fn_token_len,
                tokenizer.pad_token_id,
            )

            (predict_next_token_compiled, update_fn1!, update_fn2!) = cache_and_retrieve!(
				"model",
                compile_cache,
                cur_compiled_fn_token_len,
                model,
                padded_token_ids,
                input_mask_len,
                ps,
                st,
                next_token,
                cur_num_tokens_traced,
            )
            compile_time += time() - compile_start_time
        end

        next_token, st = predict_next_token_compiled(
            model, padded_token_ids, input_mask_len, ps, st
        )

        ntokens_generated += 1

        next_token_jl = vec(Array(next_token))

        if tokenizer.eos_token_id !== nothing &&
            all(next_token_jl .== tokenizer.eos_token_id)
            break
        end

        print(decode(tokenizer, next_token_jl))

        if cur_num_tokens >= max_context_length
            update_fn2!(padded_token_ids, next_token)
        elseif new_compiled_fn_token_len > cur_num_tokens
            input_mask_len, cur_num_tokens_traced = update_fn1!(
                padded_token_ids, input_mask_len, cur_num_tokens_traced, next_token
            )
        else
            cur_num_tokens_traced = compile_cache.increment_fn(cur_num_tokens_traced)
        end
        cur_num_tokens += 1
    end
    total_time = time() - start_time

    println()
    return ntokens_generated / (total_time - compile_time)
end
end

# ╔═╡ a0e71db2-505c-4a9d-b138-c1224bbdfed2
function gen(prompt, model, ps, st; tokenizer = smtok)

	rdev = reactant_device(; force=true)

	model = model |> rdev
	ps = ps |> rdev
	st = st |> rdev
	tokens_per_second = generate_text(model, prompt, ps, st, 100_000, tokenizer)

	println("\nTokens per second: $tokens_per_second\n\n")
	
    return nothing
end

# ╔═╡ c8734073-ae10-44ee-8a14-11de9b519688
function AttnBlock(gqa, mlp)
	layernorm(emb_dim::Int) = AlternatePrecision{Float32}(RMSNorm(emb_dim; affine = true))

	head_dim = cfg.hidden_size ÷ cfg.num_attention_heads
	
	@compact(;
		gqa, mlp, 
		ln1 = layernorm(cfg.hidden_size),
		ln2 = layernorm(cfg.hidden_size)	
	) do x

		res = x
		x = ln1(x)
		x = gqa(x)
		x = x .+ res

		res = x
		x = ln2(x)
		x = mlp(x)
		x = x .+ res

		@return x
	end
end

# ╔═╡ ff66a307-b71a-4ac1-962f-034af18d6b7f
function get_weights(weights_dict, k; permute = false)

	w = weights_dict[k]
	if permute
        w = permutedims(w, Tuple(reverse(1:ndims(w))))
    end

	w
end

# ╔═╡ c51d13e0-4dff-466f-ac36-8f0c66f18d4d
function load_weights(weights_dict, cfg)
	
	function W(num, name)
		get_weights(weights_dict, "model.layers.$num.$name.weight")
	end
	
	emb = (; weight = get_weights(weights_dict, "model.embed_tokens.weight"; 
								  permute=true))

	layers = []
	for l in 0:cfg.num_hidden_layers-1

		layer = (;
				gqa = merge(
					NamedTuple(					
						k => (; weight = W(l, "self_attn.$k")) for k in 
								(:q_proj, :k_proj, :v_proj, :o_proj)
					),
					NamedTuple(n => (; weight = ()) for n in (:q_norm, :k_norm))
				),
				mlp = (;
					proj = (; gate_proj = (;weight = W(l, "mlp.gate_proj")),
							up_proj = (; weight = W(l, "mlp.up_proj"))
						   ),
					down_proj = (; weight = W(l, "mlp.down_proj"))
				),
				ln1 = (; scale = W(l, "input_layernorm")),
				ln2 = (; scale = W(l, "post_attention_layernorm")) 
			)
		push!(layers, layer)
	end	

	layers = NamedTuple(Symbol(i) => l for (i, l) in zip(1:cfg.num_hidden_layers, layers)) 

	norm = (; scale = get_weights(weights_dict, "model.norm.weight"))

	lm_head = (; weight = get_weights(weights_dict, 
									  "model.embed_tokens.weight"))
	emb = (; weight = transpose(lm_head.weight))

	(; emb, layers, norm, lm_head)
end

# ╔═╡ 7543b520-f460-4dfd-b22d-11892a2cf3fc
ps = load_weights(weights_dict, cfg) 

# ╔═╡ 6a861eb2-6dac-4e91-84a6-be5087b35ad8
GC.gc()

# ╔═╡ 2bb9b09b-792c-47fa-aa22-59b3ec198fbc
keys(weights_dict) |> collect |> x -> filter(x) do el
	match(r"model\.layers\.11", el) == nothing
end

# ╔═╡ 35230302-0e72-44a1-aef9-dc6d5b5bd092
begin 
	getsz(n) = n
	getsz(m, n) = m * n
	getsz(t::Tuple) = getsz(t...)
end

# ╔═╡ ef42b028-0466-4e17-b1d3-ca5ad39a9ca6
[
(getsz ∘ size)(v) for v in values(weights_dict)
] |> sum

# ╔═╡ 40732463-29c3-4793-bda3-fb60a87209f9


# ╔═╡ c2285ab4-1e1e-4f90-9e77-0989f1c4bef0
function MLP(cfg)
    return Chain(;
        proj=Parallel(
            .*;
            gate_proj=Dense(cfg.hidden_size => cfg.intermediate_size, swish; use_bias=cfg.mlp_bias), # silu is just swish here
            up_proj=Dense(cfg.hidden_size => cfg.intermediate_size; use_bias=cfg.mlp_bias),
        ),
        down_proj=Dense(cfg.intermediate_size => cfg.hidden_size  ; use_bias=cfg.mlp_bias),
        name="MLP",
    )
end

# ╔═╡ 4ccc5886-ce34-4caf-9429-7521fe2eb7eb
function smol(cfg)
	
	layernorm(emb_dim::Int) = AlternatePrecision{Float32}(RMSNorm(emb_dim; affine = true))
	
	layers = map(1:32) do _
		AttnBlock(GroupedQueryAttention(cfg), MLP(cfg))
	end
	
	@compact(;
		emb = Embedding(cfg.vocab_size => cfg.hidden_size),
		layers,
		norm = layernorm(cfg.hidden_size),
		lm_head = Dense(cfg.hidden_size, cfg.vocab_size; use_bias = false)
	) do x

		x = emb(x)

		for layer in layers
			x = layer(x)
		end

		x = norm(x)
		x = lm_head(x)

		@return x
	end
end

# ╔═╡ 614b5467-3855-4949-84a3-01a964fc078d
mm = smol(cfg)

# ╔═╡ 13ba9506-f59c-44c3-916f-69c45b697654
rng = Random.default_rng()

# ╔═╡ 1f1276c4-67c0-45ca-8f43-d8d5142270fd
st = Lux.initialstates(rng, mm) 

# ╔═╡ 74116f6c-5461-411b-870a-edd5276668f8
gen("hello?", mm, ps, st)

# ╔═╡ 5fd5fb98-27c6-48aa-977d-90dfc20620dc
pps = Lux.initialparameters(rng, mm)

# ╔═╡ 6942be76-01cd-4abe-84ef-c3ffaea753b8
let
function GroupedQueryAttention(cfg)

	d_in = cfg.hidden_size
	num_heads = cfg.num_attention_heads
	num_kv_groups = cfg.num_key_value_heads

	max_pos = cfg.max_position_embeddings
    θ = cfg.rope_theta
	
    @assert num_heads % num_kv_groups == 0 
	@assert d_in % num_heads == 0 "`d_in` must be divisible by `num_heads`"
                                       
    head_dim = d_in ÷ num_heads
    d_out = num_heads * head_dim


	@compact(;
		q_proj = Dense(d_in, d_out; use_bias=false),
        k_proj = Dense(d_in, num_kv_groups * head_dim; use_bias=false),
        v_proj = Dense(d_in, num_kv_groups * head_dim; use_bias=false),
        o_proj = Dense(d_out, d_in; use_bias=false),
        q_norm = altRMSNorm(head_dim),
        k_norm = altRMSNorm(head_dim),
		cache = @init_fn(rng -> compute_rotary_embedding_params(
        head_dim, max_pos; base=θ, dtype=Float32
    ), :state)
	) do x

		_, num_tokens, batch = size(x)
		
		queries = q_proj(x) 
		keys = k_proj(x)
		values = v_proj(x)

		#reshape to 4 dims
		queries = reshape(queries, head_dim, num_heads, num_tokens, batch)
    	keys = reshape(keys, head_dim, num_kv_groups, num_tokens, batch)
    	values = reshape(values, head_dim, num_kv_groups, num_tokens, batch)

		queries = q_norm(queries)
		keys = k_norm(keys)

		
		queries = apply_rope(queries, cache[:cos_cache], cache[:sin_cache])
		keys = apply_rope(keys, cache[:cos_cache], cache[:sin_cache])

		context = scaled_dot_product_attention(
			queries, keys, values;
		)[1]

		context = reshape(context,head_dim * num_heads, num_tokens, batch)

		proj = o_proj(context)
		
		@return proj
	end
end

gqa = GroupedQueryAttention(cfg)

ps, st = Lux.setup(rng, gqa)

x = rand(Float32, (960, 7, 1))

gqa(x, ps, st)[1]
end

# ╔═╡ b87f6127-4812-4e78-ad2f-5fcfdb792391
function generate(model, ps, st, tokenizer, prompt; n_tokens=30)
	
    tokens = encode(tokenizer, prompt)
 
    @progress for _ in 1:n_tokens
		o, _ = model(reshape(tokens, :, 1), ps, st)
        next_token = argmax(o[:, end, 1])
        push!(tokens, next_token)
    end
    decode(tokenizer, tokens .- 1)
end

# ╔═╡ 60843222-91b2-4d0d-82df-26b2136b8231
function load_model_and_gen(get_model, cfg, tokenizer, prompt; n_token = 3)
	gpu = gpu_device()
	mm = get_model(cfg)
	ps = load_weights(weights_dict, cfg) |> gpu
	st = Lux.initialstates(rng, mm)  |> gpu

	generate(mm, ps, st, tokenizer, prompt; n_token)
end

# ╔═╡ e2ee0274-5847-440d-9f1d-c2117d63b256


# ╔═╡ 4f7df351-2f13-4e24-9fa8-4cfbc45fdc25


# ╔═╡ 22cb4854-8143-4631-ac30-7f89bab7541d


# ╔═╡ 9602e97b-ed3e-4f21-880e-68656a2b50ff


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
ConcreteStructs = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
HuggingFaceTokenizers = "a6888d44-1185-43bb-bd0f-7806f9976d18"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
SafeTensors = "eeda0dda-7046-4914-a807-2495fc7abb89"

[compat]
BFloat16s = "~0.5.1"
ConcreteStructs = "~0.2.3"
HuggingFaceTokenizers = "~0.1.0"
JSON3 = "~1.14.3"
Lux = "~1.29.4"
ProgressLogging = "~0.1.6"
PythonCall = "~0.9.31"
Reactant = "~0.2.203"
SafeTensors = "~1.2.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.5"
manifest_format = "2.0"
project_hash = "9881a4010585a569760c8263cf9e4cefac447760"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "d81ae5489e13bc03567d4fbbb06c546a5e53c857"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.22.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "3b642331600250f592719140c60cf12372b82d66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BufferedStreams]]
git-tree-sha1 = "6863c5b7fc997eadcabdbaf6c5f201dc30032643"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.2"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "Preferences", "Scratch", "TOML", "pixi_jll"]
git-tree-sha1 = "bd491d55b97a036caae1d78729bdb70bf7dababc"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.33"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DLFP8Types]]
git-tree-sha1 = "422996f4ddf0ed17a9a3b729347fcbf38fae80b2"
uuid = "f4c16678-4a16-415b-82ef-ed337c5d6c7c"
version = "0.1.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "fc5e4798a96c4a7ffa2991bb5b372262578587a9"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.27"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnumX]]
git-tree-sha1 = "7bebc8aad6ee6217c78c5ddcf7ed289d65d0263e"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.6"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "2e255d8a2944ae78ed0fa48ea6570bdc964073f4"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.118"
weakdeps = ["ADTypes", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "LogExpFunctions", "SpecialFunctions", "StaticArrays"]

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeGPUArraysCoreExt = "GPUArraysCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

[[deps.EnzymeCore]]
git-tree-sha1 = "990991b8aa76d17693a98e3a915ac7aa49f08d1a"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.18"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "97d6bfda1bd724fdf2930e66bd74a4b99b1cc822"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.238+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "4a8c0a9eebf807ac42f0f6de758e60a20be25ffb"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.3"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "b2977f86ed76484de6f29d5b36f2fa686f085487"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "7237207d87760307c92c260565f706688229cf6e"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.8.1"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HuggingFaceTokenizers]]
deps = ["PythonCall"]
git-tree-sha1 = "7d1211d86e5d75ab68e0ad0c2cc83feec0d8b1e3"
uuid = "a6888d44-1185-43bb-bd0f-7806f9976d18"
version = "0.1.0"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b5a371fcd1d989d844a4354127365611ae1e305f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.39"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "ce8614210409eaa54ed5968f4b50aa96da7ae543"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.4"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.Lux]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DiffResults", "DispatchDoctor", "EnzymeCore", "FastClosures", "ForwardDiff", "Functors", "GPUArraysCore", "LinearAlgebra", "LuxCore", "LuxLib", "MLDataDevices", "MacroTools", "Markdown", "NNlib", "Optimisers", "PrecompileTools", "Preferences", "Random", "ReactantCore", "Reexport", "SciMLPublic", "Setfield", "Static", "StaticArraysCore", "Statistics", "UUIDs", "WeightInitializers"]
git-tree-sha1 = "5d38327c99da895e67f81131eb325c5ad170761f"
uuid = "b2108857-7c20-44ae-9111-449ecde12c47"
version = "1.29.4"

    [deps.Lux.extensions]
    ComponentArraysExt = "ComponentArrays"
    EnzymeExt = "Enzyme"
    FluxExt = "Flux"
    GPUArraysExt = "GPUArrays"
    LossFunctionsExt = "LossFunctions"
    MLUtilsExt = "MLUtils"
    MPIExt = "MPI"
    MPINCCLExt = ["CUDA", "MPI", "NCCL"]
    MooncakeExt = "Mooncake"
    ReactantExt = ["Enzyme", "Reactant", "Functors", "ReactantCore", "LuxLib", "MLDataDevices", "Optimisers", "NNlib", "Statistics"]
    ReverseDiffExt = ["FunctionWrappers", "ReverseDiff"]
    SimpleChainsExt = "SimpleChains"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"

    [deps.Lux.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    FunctionWrappers = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SimpleChains = "de6bee2f-e2f4-4ec7-b6ed-219cc6f6e9e5"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxCore]]
deps = ["DispatchDoctor", "Random", "SciMLPublic"]
git-tree-sha1 = "9455b1e829d8dacad236143869be70b7fdb826b8"
uuid = "bb33d45b-7691-41d6-9220-0943567d0623"
version = "1.5.3"

    [deps.LuxCore.extensions]
    ArrayInterfaceReverseDiffExt = ["ArrayInterface", "ReverseDiff"]
    ArrayInterfaceTrackerExt = ["ArrayInterface", "Tracker"]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    FluxExt = "Flux"
    FunctorsExt = "Functors"
    MLDataDevicesExt = ["Adapt", "MLDataDevices"]
    ReactantExt = "Reactant"
    SetfieldExt = "Setfield"

    [deps.LuxCore.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.LuxLib]]
deps = ["ArrayInterface", "CPUSummary", "ChainRulesCore", "DispatchDoctor", "EnzymeCore", "FastClosures", "Functors", "KernelAbstractions", "LinearAlgebra", "LuxCore", "MLDataDevices", "Markdown", "NNlib", "Preferences", "Random", "Reexport", "SciMLPublic", "Static", "StaticArraysCore", "Statistics", "UUIDs"]
git-tree-sha1 = "bc4984518909e9c0bf78cc8bb837cb63534ab82c"
uuid = "82251201-b29d-42c6-8e01-566dec8acb11"
version = "1.15.2"

    [deps.LuxLib.extensions]
    AppleAccelerateExt = "AppleAccelerate"
    BLISBLASExt = "BLISBLAS"
    CUDAExt = "CUDA"
    CUDAForwardDiffExt = ["CUDA", "ForwardDiff"]
    EnzymeExt = "Enzyme"
    ForwardDiffExt = "ForwardDiff"
    LoopVectorizationExt = ["LoopVectorization", "Polyester"]
    MKLExt = "MKL"
    OctavianExt = ["Octavian", "LoopVectorization"]
    ReactantExt = ["Reactant", "ReactantCore"]
    ReverseDiffExt = "ReverseDiff"
    SLEEFPiratesExt = "SLEEFPirates"
    TrackerAMDGPUExt = ["AMDGPU", "Tracker"]
    TrackerExt = "Tracker"
    cuDNNExt = ["CUDA", "cuDNN"]

    [deps.LuxLib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AppleAccelerate = "13e28ba4-7ad8-5781-acae-3021b1ed3924"
    BLISBLAS = "6f275bd8-fec0-4d39-945b-7e95a765fa1e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
    MKL = "33e6dc65-8f57-5167-99aa-e5a354878fb2"
    Octavian = "6fd5a793-0b7e-452c-907f-f8bfe9c57db4"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReactantCore = "a3311ec8-5e00-46d5-b541-4f83e724a433"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SLEEFPirates = "476501e8-09a2-5ece-8869-fb82de89a1fa"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "117dd3d538d0ca82979ebcf15d9ad8bf0431c206"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.2"

    [deps.MLDataDevices.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    ChainRulesExt = "ChainRules"
    ComponentArraysExt = "ComponentArrays"
    FillArraysExt = "FillArrays"
    GPUArraysSparseArraysExt = ["GPUArrays", "SparseArrays"]
    MLUtilsExt = "MLUtils"
    MetalExt = ["GPUArrays", "Metal"]
    OneHotArraysExt = "OneHotArrays"
    OpenCLExt = ["GPUArrays", "OpenCL"]
    ReactantExt = "Reactant"
    RecursiveArrayToolsExt = "RecursiveArrayTools"
    ReverseDiffExt = "ReverseDiff"
    SparseArraysExt = "SparseArrays"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"
    cuDNNExt = ["CUDA", "cuDNN"]
    oneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "011cab361eae7bcd7d278f0a7a00ff9c69000c51"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.14"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "6dc9ffc3a9931e6b988f913b49630d0fb986d0a8"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.33"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "22faba70c22d2f03e60fbc61da99c4ebfc3eb9ba"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.5.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "36b5d2b9dd06290cd65fcf5bdbc3a551ed133af5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.7"
weakdeps = ["Adapt", "EnzymeCore", "Reactant"]

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "c5a07210bd060d6a8491b0ccdee2fa0235fc00bf"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.ProtoBuf]]
deps = ["BufferedStreams", "Dates", "EnumX", "TOML"]
git-tree-sha1 = "eabdb811dbacadc9d7e0dee9ac11c1a12705e12a"
uuid = "3349acd9-ac6a-5e09-bcdb-63829b23a429"
version = "1.2.0"

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "982f3f017f08d31202574ef6bdcf8b3466430bea"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.31"

    [deps.PythonCall.extensions]
    CategoricalArraysExt = "CategoricalArrays"
    PyCallExt = "PyCall"

    [deps.PythonCall.weakdeps]
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reactant]]
deps = ["Adapt", "CEnum", "Crayons", "Downloads", "EnumX", "Enzyme", "EnzymeCore", "Functors", "GPUArraysCore", "GPUCompiler", "HTTP", "JSON3", "LLVM", "LLVMOpenMP_jll", "Libdl", "LinearAlgebra", "OrderedCollections", "PrecompileTools", "Preferences", "PrettyTables", "ProtoBuf", "Random", "ReactantCore", "Reactant_jll", "ScopedValues", "Scratch", "Setfield", "Sockets", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "9fdfad47cca824ae1941e96c01b2e7c6db7e0e09"
uuid = "3c362404-f566-11ee-1572-e11a4b42c853"
version = "0.2.203"

    [deps.Reactant.extensions]
    ReactantAbstractFFTsExt = "AbstractFFTs"
    ReactantArrayInterfaceExt = "ArrayInterface"
    ReactantCUDAExt = ["CUDA", "GPUCompiler", "KernelAbstractions", "LLVM"]
    ReactantDLFP8TypesExt = "DLFP8Types"
    ReactantFFTWExt = ["FFTW", "AbstractFFTs", "LinearAlgebra"]
    ReactantFillArraysExt = "FillArrays"
    ReactantFloat8sExt = "Float8s"
    ReactantKernelAbstractionsExt = "KernelAbstractions"
    ReactantMPIExt = "MPI"
    ReactantNNlibExt = ["NNlib", "Statistics"]
    ReactantNPZExt = "NPZ"
    ReactantOffsetArraysExt = "OffsetArrays"
    ReactantOneHotArraysExt = "OneHotArrays"
    ReactantPythonCallExt = "PythonCall"
    ReactantRandom123Ext = "Random123"
    ReactantSparseArraysExt = "SparseArrays"
    ReactantSpecialFunctionsExt = "SpecialFunctions"
    ReactantStatisticsExt = "Statistics"
    ReactantYaoBlocksExt = "YaoBlocks"
    ReactantZygoteExt = "Zygote"

    [deps.Reactant.weakdeps]
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    DLFP8Types = "f4c16678-4a16-415b-82ef-ed337c5d6c7c"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Float8s = "81dfefd7-55b0-40c6-a251-db853704e186"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    Random123 = "74087812-796a-5b5d-8853-05524746bad3"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    YaoBlocks = "418bc28f-b43b-5e0b-a6e7-61bbc1a2c1df"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.ReactantCore]]
deps = ["ExpressionExplorer", "MacroTools"]
git-tree-sha1 = "f3e31b90afcd152578a6c389eae46dd38b9a4f38"
uuid = "a3311ec8-5e00-46d5-b541-4f83e724a433"
version = "0.1.16"

[[deps.Reactant_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "4b3f53d3c8c17534522778095223d34c3b9c0d12"
uuid = "0192cb87-2b54-54ad-80e0-3be72ad8a3c0"
version = "0.0.305+0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SafeTensors]]
deps = ["BFloat16s", "DLFP8Types", "JSON3", "MappedArrays", "Mmap", "ProgressMeter"]
git-tree-sha1 = "87dd645cd717affdb9bcd9a47563680d3ca997e3"
uuid = "eeda0dda-7046-4914-a807-2495fc7abb89"
version = "1.2.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "49440414711eddc7227724ae6e570c7d5559a086"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.3.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eee1b9ad8b29ef0d936e3ec9838c7ec089620308"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.16"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a3c1536470bf8c5e02096ad4853606d7c8f62721"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.2"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.WeightInitializers]]
deps = ["ConcreteStructs", "GPUArraysCore", "LinearAlgebra", "Random", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d79b71da9e7be904db615bdb99187d30753822a4"
uuid = "d49dbf32-c5c2-4618-8acc-27bb2598ef2d"
version = "1.3.1"

    [deps.WeightInitializers.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    GPUArraysExt = "GPUArrays"
    ReactantExt = "Reactant"

    [deps.WeightInitializers.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "2ca2ac0b23a8e6b76752453e08428b3b4de28095"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.5.12+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.pixi_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "f349584316617063160a947a82638f7611a8ef0f"
uuid = "4d7b5844-a134-5dcd-ac86-c8f19cd51bed"
version = "0.41.3+0"
"""

# ╔═╡ Cell order:
# ╠═5c0a8852-087d-11f1-950b-cdf741336217
# ╠═b7fdbfd6-a3a7-402d-bb27-f4f8258ff7fa
# ╠═63b217ae-3900-47ee-be17-bcabc038858a
# ╠═1f1276c4-67c0-45ca-8f43-d8d5142270fd
# ╠═7543b520-f460-4dfd-b22d-11892a2cf3fc
# ╠═5fd5fb98-27c6-48aa-977d-90dfc20620dc
# ╠═18606732-3cbd-4e15-8ff3-323181368342
# ╠═1a0285af-a049-4c79-955b-a8c4ff7e3f90
# ╠═0f6feaa5-3d27-4a40-888d-e49f2d071e9c
# ╠═a0e71db2-505c-4a9d-b138-c1224bbdfed2
# ╠═74116f6c-5461-411b-870a-edd5276668f8
# ╠═a7c4f7ac-701b-4f05-8132-3dfec70585c7
# ╠═27f087ae-876b-4a7d-920d-1d507f47d294
# ╠═60843222-91b2-4d0d-82df-26b2136b8231
# ╠═db7b31c7-7cea-443a-9107-b0f0fe44967a
# ╠═d10633a6-44dd-4a9a-9881-f36a15a8e57d
# ╠═d8a6c364-ef26-4b70-ab68-72c46ae2c7ec
# ╠═9999bb0f-2d52-4777-9334-6c2e58141721
# ╠═eb64305e-281b-4cf4-9944-22aed864f66e
# ╠═9bcb4599-e27b-48c1-a2f0-78d4e1972888
# ╠═c8734073-ae10-44ee-8a14-11de9b519688
# ╠═6942be76-01cd-4abe-84ef-c3ffaea753b8
# ╠═2e05fefb-ed47-4e51-b704-bc744c3a7fbe
# ╠═4ccc5886-ce34-4caf-9429-7521fe2eb7eb
# ╠═ff66a307-b71a-4ac1-962f-034af18d6b7f
# ╠═614b5467-3855-4949-84a3-01a964fc078d
# ╠═c51d13e0-4dff-466f-ac36-8f0c66f18d4d
# ╠═6a861eb2-6dac-4e91-84a6-be5087b35ad8
# ╠═2bb9b09b-792c-47fa-aa22-59b3ec198fbc
# ╠═ef42b028-0466-4e17-b1d3-ca5ad39a9ca6
# ╠═35230302-0e72-44a1-aef9-dc6d5b5bd092
# ╠═40732463-29c3-4793-bda3-fb60a87209f9
# ╠═c2285ab4-1e1e-4f90-9e77-0989f1c4bef0
# ╠═13ba9506-f59c-44c3-916f-69c45b697654
# ╠═b87f6127-4812-4e78-ad2f-5fcfdb792391
# ╠═804f121f-6c51-4e41-b366-c4410cc05ac8
# ╠═e2ee0274-5847-440d-9f1d-c2117d63b256
# ╠═4f7df351-2f13-4e24-9fa8-4cfbc45fdc25
# ╠═22cb4854-8143-4631-ac30-7f89bab7541d
# ╠═9602e97b-ed3e-4f21-880e-68656a2b50ff
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
