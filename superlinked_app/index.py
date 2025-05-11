from openai import OpenAI
from superlinked_app import constants
from superlinked import framework as sl

client = OpenAI()

def openai_embed(text: str) -> list[float]:
    resp = client.embeddings.create(
                                input=text,
                                model="text-embedding-3-small"          # <-- your desired model here
                                )
    return resp.data[0].embedding

class ProductSchema(sl.Schema):
        id: sl.IdField
        type: sl.String
        category: sl.StringList

        title: sl.String
        description: sl.String

        # these fields will hold your precomputed embeddings
        title_embedding: sl.FloatList
        description_embedding: sl.FloatList

        review_rating: sl.Float
        review_count: sl.Integer
        price: sl.Float

product = ProductSchema()

category_space = sl.CategoricalSimilaritySpace(
                                            category_input=product.category,
                                            categories=constants.CATEGORIES,
                                            uncategorized_as_category=True,
                                            negative_filter=-1,
                                            )

# title_space = sl.TextSimilaritySpace(
#                                     text=product.title, 
#                                     model="Alibaba-NLP/gte-large-en-v1.5"
#                                     )

# description_space = sl.TextSimilaritySpace(
#                                     text=product.description, 
#                                     model="Alibaba-NLP/gte-large-en-v1.5"
#                                     )

title_space = sl.CustomSpace(
                            vector=product.title_embedding,
                            length=1536,
                            description="Embedding from OpenAI text-embedding-3-small"
                            )

description_space = sl.CustomSpace(
                                vector=product.description_embedding,
                                length=1536,
                                description="Embedding from OpenAI text-embedding-3-small"
                                )

review_rating_maximizer_space = sl.NumberSpace(
                                            number=product.review_rating, 
                                            mode=sl.Mode.MAXIMUM,
                                            min_value=-1.0, 
                                            max_value=5.0
                                            )

price_minimizer_space = sl.NumberSpace(
                                    number=product.price, 
                                    mode=sl.Mode.MINIMUM, 
                                    min_value=0.0, 
                                    max_value=1000
                                    )

product_index = sl.Index(
                        spaces=[
                                title_space,
                                description_space,
                                review_rating_maximizer_space,
                                price_minimizer_space,
                                ],
                        fields=[
                                product.type, 
                                product.category, 
                                product.review_rating, 
                                product.price
                                ],
                        )