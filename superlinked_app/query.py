from superlinked import framework as sl
from superlinked_app import constants, index
from superlinked_app.config import settings

assert (
        settings.OPENAI_API_KEY
        ), "OPENAI_API_KEY must be set in environment variables to use natural language queries"


openai_config = sl.OpenAIClientConfig(
                                    api_key=settings.OPENAI_API_KEY.get_secret_value(), 
                                    model=settings.OPENAI_MODEL_ID
                                    )


title_similar_param = sl.Param(
                                "query_title",
                                description=(
                                    "The text in the user's query that is used to search in the products' title."
                                    "Extract info that does not apply to other spaces or params."
                                ),
                            )

text_similar_param = sl.Param(
                            "query_description",
                            description=(
                                "The text in the user's query that is used to search in the products' description."
                                " Extract info that does not apply to other spaces or params."
                            ),
                            )

'''
The here to understand is that their are two types of variables that we can use for the search 

    1. Rule based filtering : These are mainly numeric values (eg : price, review rating) & categorical values (eg : category, type)
    2. Semantic search : These are the text values (eg : title, description)

we use,

    .filter() : for rule based filtering
    .with_vector() : for vector search
    .similar() : for semantic search
    
'''
base_query = (
            sl.Query(
                index.product_index,
                weights={
                        index.title_space: sl.Param("title_weight"),
                        index.description_space: sl.Param("description_weight"),
                        index.review_rating_maximizer_space: sl.Param(
                            "review_rating_maximizer_weight"
                            ),
                        index.price_minimizer_space: sl.Param("price_minimizer_weights"),
                        },
            )
            .find(index.product)
            .limit(sl.Param("limit"))
            .with_natural_query(sl.Param("natural_query"), openai_config)
            .filter(
                index.product.type
                == sl.Param(
                    "filter_by_type",
                    description="Used to only present items that have a specific type",
                    options=constants.TYPES,
                )
            )
        )

filter_query = (
                base_query.similar(
                    index.description_space,
                    text_similar_param,
                    sl.Param("description_similar_clause_weight"),
                )
                .filter(
                    index.product.category
                    == sl.Param(                                                    ## Equal Filter
                        "filter_by_cateogry",
                        description="Used to only present items that have a specific cateogry",
                        options=constants.CATEGORIES,
                    )
                )
                .filter(
                    index.product.review_rating                                     ## Greater than Filter
                    >= sl.Param(
                        "review_rating_bigger_than",
                        description="Used to find items with a review rating bigger than the provided number.",
                    )
                )
                .filter(
                    index.product.price
                    <= sl.Param(                                                    ## Less than Filter
                        "price_smaller_than",
                        description="Used to find items with a price smaller than the provided number.",
                    )
                )
)

# leverages the full power of vector spaces to understand and match complex search intentions across multiple attributes
semantic_query = (
                base_query.similar(
                    index.description_space,
                    text_similar_param,
                    sl.Param("description_similar_clause_weight"),
                )
                .similar(
                    index.title_space,
                    title_similar_param,
                    sl.Param("title_similar_clause_weight"),
                )
                .filter(
                    index.product.category
                    == sl.Param(
                        "filter_by_cateogry",
                        description="Used to only present items that have a specific cateogry",
                        options=constants.CATEGORIES,
                    )
                )
)

# enabling product recommendations and "more like this"
similar_items_query = semantic_query.with_vector(index.product, sl.Param("product_id"))