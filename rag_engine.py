import os
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ✅ OpenAI v1+ client (reads OPENAI_API_KEY automatically, but we validate)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or export it.")

client = OpenAI(api_key=api_key)

# ---- Abuja Luxury Properties Knowledge Base ----
PROPERTIES = [
    # Maitama Properties
    "Presidential Villa-Style Mansion in Maitama - 8 bedrooms, 10 bathrooms, 15,000 sqft, ₦2.5B. Features: Swimming pool, helipad, wine cellar, home theater, smart home automation.",
    "Gated Estate Mansion in Maitama - 6 bedrooms, 7 bathrooms, 11,000 sqft, ₦1.5B. Features: Gated community, tennis court, gazebo, fountain, premium security.",
    "Ultimate Luxury Estate in Maitama - 10 bedrooms, 12 bathrooms, 18,000 sqft, ₦3.2B. Features: Indoor pool, private cinema, spa, guest villa, 8-car garage.",
    "Premium Garden Residence in Maitama - 6 bedrooms, 6 bathrooms, 9,500 sqft, ₦1.35B. Features: Botanical garden, koi pond, guest suite, home theater.",
    # Asokoro Properties
    "Modern Hilltop Residence in Asokoro - 6 bedrooms, 7 bathrooms, 12,000 sqft, ₦1.8B. Features: Panoramic city views, elevator, solar power, guest house.",
    "Hillside Contemporary Villa in Asokoro - 5 bedrooms, 6 bathrooms, 9,800 sqft, ₦1.1B. Features: City views, infinity pool, home gym, solar panels.",
    "Luxury Residential Complex in Asokoro - 5 bedrooms, 5 bathrooms, 8,800 sqft, ₦920M. Features: Community pool, security gate, fitness center, tennis court.",
    # Wuse II Properties
    "Diplomatic Luxury Apartment in Wuse II - 5 bedrooms, 5 bathrooms, 8,500 sqft penthouse, ₦850M. Features: Rooftop garden, concierge, gym, underground parking.",
    "Luxury Duplex Apartment in Wuse II - 3 bedrooms, 3 bathrooms, 5,500 sqft, ₦450M. Features: Private balcony, modern kitchen, smart home, 24/7 security.",
    "Exclusive Central Mansion in Wuse II - 6 bedrooms, 6 bathrooms, 10,200 sqft, ₦1.1B. Features: Central business district, modern design, smart automation.",
    # Garki II Properties
    "Executive Townhouse in Garki II - 4 bedrooms, 4 bathrooms, 6,500 sqft, ₦650M. Features: Private garden, study room, staff quarters, security system.",
    "Executive Garden Apartment in Garki II - 3 bedrooms, 3 bathrooms, 4,800 sqft, ₦380M. Features: Private garden, covered parking, concierge, play area.",
    # Guzape Properties
    "Contemporary Designer Home in Guzape - 5 bedrooms, 5 bathrooms, 7,200 sqft, ₦750M. Features: Smart home system, home office, EV charging, landscaped garden.",
    "Modern Architectural Masterpiece in Guzape - 4 bedrooms, 4 bathrooms, 6,800 sqft, ₦680M. Features: Open plan design, rooftop terrace, home office, smart lighting.",
    "Modern Minimalist Estate in Guzape - 5 bedrooms, 5 bathrooms, 8,400 sqft, ₦890M. Features: Clean lines, large windows, green roof, modern kitchen.",
    # Jabi Properties
    "Waterfront Luxury Villa in Jabi - 7 bedrooms, 8 bathrooms, 10,500 sqft, ₦1.2B. Features: Private dock, infinity pool, boat house, guest cottage.",
    "Lakefront Luxury Residence in Jabi - 4 bedrooms, 5 bathrooms, 8,200 sqft, ₦850M. Features: Lake views, private beach, boat slip, outdoor kitchen.",
    "Premium Waterside Villa in Jabi - 5 bedrooms, 6 bathrooms, 9,200 sqft, ₦1.05B. Features: Lake access, sunset views, large deck, water sports.",
    # Utako Properties
    "Skyline Penthouse in Utako - 5 bedrooms, 5 bathrooms, 9,000 sqft, ₦950M. Features: 360-degree views, private elevator, wine room, smart glass.",
    "Business Executive Apartment in Utako - 3 bedrooms, 3 bathrooms, 5,200 sqft, ₦420M. Features: Business center, meeting rooms, high-speed internet, secure parking.",
]

NEIGHBORHOODS = [
    "Maitama: Abuja's most prestigious diplomatic neighborhood with wide streets, manicured gardens, exclusive security, and luxury estates. Home to embassies and high-net-worth individuals.",
    "Asokoro: Exclusive hillside residential area known for panoramic city views, proximity to Presidential Villa, and luxury mansions for top officials and executives.",
    "Wuse II: Upscale commercial and residential district with luxury apartments, fine dining, high-end shopping, and international schools for expatriates.",
    "Garki II: Prime commercial district transitioning into luxury residential area with modern apartments, international schools, and premium amenities.",
    "Guzape: Emerging luxury district with contemporary architecture, gated communities, modern amenities, popular with young professionals and diplomats.",
    "Jabi: Lakeside luxury area featuring waterfront properties, modern apartments, proximity to Jabi Lake Mall and recreational facilities.",
    "Utako: Modern business and residential district with luxury apartments, office complexes, upscale shopping centers, and excellent connectivity.",
]

MARKET_CONTEXT = [
    "Abuja property market: Premium real estate in elite neighborhoods ranging from ₦380M to ₦3.2B.",
    "Investment trends: Diplomatic areas like Maitama maintain stable values, emerging districts like Guzape offer growth potential.",
    "Luxury features: Smart home automation, private security, swimming pools, home theaters, wine cellars, guest houses.",
    "Property types: Luxury villas, diplomatic residences, penthouses, waterfront estates, modern apartments, gated community homes.",
]

FULL_KNOWLEDGE_BASE = PROPERTIES + NEIGHBORHOODS + MARKET_CONTEXT

# ---- Create Embeddings (OpenAI v1+) ----
def embed_texts(texts, model: str = "text-embedding-3-small"):
    """
    Create embeddings for given texts using OpenAI v1+.
    Returns float32 numpy array of shape (len(texts), dim).
    """
    try:
        resp = client.embeddings.create(model=model, input=texts)
        embeddings = [item.embedding for item in resp.data]
        return np.array(embeddings, dtype="float32")
    except Exception as e:
        print(f"Embedding error: {e}")
        # fallback (keeps app running but quality is random)
        return np.random.randn(len(texts), 1536).astype("float32")

# Build FAISS index
try:
    property_embeddings = embed_texts(FULL_KNOWLEDGE_BASE)
    dim = property_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(property_embeddings)
    print(f"FAISS index created with {len(FULL_KNOWLEDGE_BASE)} items")
except Exception as e:
    print(f"FAISS initialization error: {e}")
    index = None

def retrieve_context(query, k=3):
    """Retrieve relevant property information based on query."""
    if index is None:
        return "\n".join(PROPERTIES[:3])

    try:
        query_embedding = embed_texts([query])
        distances, indices = index.search(query_embedding, k)

        relevant_texts = []
        for idx in indices[0]:
            if 0 <= idx < len(FULL_KNOWLEDGE_BASE):
                relevant_texts.append(FULL_KNOWLEDGE_BASE[idx])

        return "\n".join(relevant_texts) if relevant_texts else "\n".join(PROPERTIES[:3])
    except Exception as e:
        print(f"Retrieval error: {e}")
        return "\n".join(PROPERTIES[:3])

def generate_with_rag(user_prompt: str):
    """Generate response using RAG with Abuja property data (OpenAI v1+)."""
    context = retrieve_context(user_prompt)

    system_prompt = (
        "You are ARIA (Abuja Real Estate Intelligence Assistant), a premium AI for Abuja luxury properties. "
        "Provide accurate, detailed information about properties, neighborhoods, and market insights. "
        "Be professional and helpful."
    )

    user_prompt_full = f"""Context Information:
{context}

User Question: {user_prompt}

Please answer based on the context above. If you don't know something, say so."""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_full},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Chat completion error: {e}")
        return (
            f"Based on Abuja property data: {user_prompt}\n\n"
            "We have luxury properties in Maitama, Asokoro, and other elite areas. "
            "Please contact our agents for specific details."
        )

def refine_with_edit(original: str, edited: str):
    """Refine response based on user edits (OpenAI v1+)."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are ARIA, a luxury real estate AI."},
                {"role": "user", "content": f"Original: {original}\n\nEdited version: {edited}\n\nPlease rewrite this to be more polished and professional."},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Refine error: {e}")
        return edited

def regenerate_section(full_text: str, selected_section: str, instruction: str):
    """Regenerate a specific section of text (OpenAI v1+)."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are ARIA, a luxury real estate AI."},
                {"role": "user", "content": f"Full text: {full_text}\n\nSection to rewrite: {selected_section}\n\nInstruction: {instruction}\n\nRewrite only that section."},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Regenerate section error: {e}")
        return full_text
