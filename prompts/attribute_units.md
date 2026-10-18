Perform attribute unit extraction for the product review provided as input.

Identify every distinct product-related attribute that the reviewer experiences, observes, or evaluates. Return each distinct attribute as one attribute unit with one `raw_attribute` and one review-grounded `sentiment`.

This is Attribute Unit extraction, not Opinion Unit extraction. Do not return `raw_aspect`, `raw_status`, `excerpt`, `opinion`, evidence, confidence, explanation, or a single forced attribute.

## Raw attribute

A `raw_attribute` is a concise, clusterable name for one product component, property, behavior, experienced consequence, or usage situation. It is a grouping key, not a summary of the reviewer's opinion.

Use a short noun phrase, preferably one to three words, such as `화면 밝기`, `게임 화면`, `잔상`, `화면 선명도`, `빛샘`, `장시간 시청`, or `무게`.

Keep evaluation, polarity, state, degree, comparison, cause, consequence details, context, time, and sentence endings out of the label.

Examples:

- use `화면 밝기`, not `너무 밝은 화면` or `화면 밝기가 마음에 듦`;
- use `잔상`, not `움직이는 화면에 잔상이 자주 남음`;
- use `빛샘`, not `심한 빛샘이 거슬림`;
- use `장시간 시청`, not `장시간 화면을 볼 때 눈이 편함`.
- use `화면 크기`, not `27인치 화면 크기` or `큰 화면`.

Name the attribute without including its observed value or evaluation. A size such as `27인치` belongs to the review evidence, not to the attribute name.

Use the same compact label for equivalent concepts. Prefer a concrete attribute over broad words such as `품질`, `성능`, `특성`, or `사용 경험` when the review supports a more specific target.

If the review explicitly evaluates only the product as a whole and no specific attribute can be identified, use `전반적 상품 경험`. Do not use it as a fallback when a specific attribute is available.

Return each underlying attribute at most once per review. When the same attribute appears repeatedly, combine the evidence internally and assign one overall sentiment. Do not join different attributes into one label.

Do not add `품질` or `전반적 상품 경험` for an evaluation already explained by a specific attribute. Include a separate whole-product evaluation only when it meets the whole-product rule above.

## Sentiment

Each `sentiment` must describe the reviewer's direction toward its paired `raw_attribute` and must be exactly one of:

- `positive`: the reviewer clearly prefers or is satisfied with the attribute or its experienced state;
- `negative`: the reviewer is clearly dissatisfied with the attribute or its experienced state;
- `mixed`: the review explicitly contains both advantages and disadvantages for that same attribute;
- `neutral`: the review states a fact or experience for the attribute without evaluating it;
- `unknown`: the evidence is insufficient or too uncertain to determine an evaluation direction.

Do not infer sentiment from the attribute name or state alone:

- a heavy monitor can be `positive` when the reviewer likes the solid feel;
- the same heavy monitor can be `negative` when it is inconvenient to move;
- a reported 4kg monitor weight without preference is `neutral`;
- `더 써봐야 알겠다` is `unknown`;
- repeated positive and negative evaluations of the same attribute become one `mixed` attribute.

If the review evaluates different attributes in different directions, return separate items with separate sentiments. Never assign one whole-review sentiment to every attribute.

A factual observation does not inherit praise for an adjacent attribute. For example, in `다양한 기능이 있는데 가격이 저렴해서 만족합니다.`, feature variety is `neutral` unless it is separately praised, while price is `positive`.

## Grounding and scope

Extract only the reviewer's own experience, direct observation, or explicit evaluation of the reviewed product.

The product name and category are context only. Use them to understand terminology and references, but never treat metadata as evidence.

Do not infer unmentioned technical causes, features, attributes, or evaluation directions. Do not return a comparison product as an attribute.

Exclude shipping, packaging, seller service, payment, delivery issues, promotions, free gifts, and content unrelated to the product or its use. A logistics-only review returns an empty array.

Coupon discounts and promotion-dependent purchase prices are excluded transaction details, even when described as cheap or satisfying. Do not turn `쿠폰을 적용해 싸게 샀어요` into a price attribute. An independently stated product-price or value-for-money evaluation elsewhere in the review may still be extracted.

Use the same language as the review for `raw_attribute`.

Return valid JSON only. Do not return Markdown, explanations, headings, code fences, or extra fields. Use exactly this format:

{
  "attribute_units": [
    {
      "raw_attribute": "속성 명칭",
      "sentiment": "positive | negative | mixed | neutral | unknown"
    }
  ]
}

If the review contains no valid product-related observation or evaluation, return `{ "attribute_units": [] }`.

## Few-shot example 1: several monitor attributes

Product name:

32인치 모니터

Product category:

모니터

Review:

<review>
이전 모니터보다 무게가 조금 무겁지만 저는 묵직해서 마음에 듭니다. 장시간 화면을 볼 때 눈이 편했고, 빛샘은 꽤 거슬렸습니다.
</review>

Output:

{
  "attribute_units": [
    {
      "raw_attribute": "무게",
      "sentiment": "positive"
    },
    {
      "raw_attribute": "장시간 시청",
      "sentiment": "positive"
    },
    {
      "raw_attribute": "빛샘",
      "sentiment": "negative"
    }
  ]
}

## Few-shot example 2: mixed, neutral, unknown, and exclusion

Product name:

게이밍 모니터

Product category:

모니터

Review:

<review>
배송은 빨랐습니다. 화면 밝기는 낮에 보기에는 충분해서 좋지만 밤에는 너무 밝아 눈부신 점이 아쉽습니다. HDMI 단자는 직접 확인해 보니 2개입니다. 받침대 내구성은 더 써봐야 알 것 같습니다.
</review>

Output:

{
  "attribute_units": [
    {
      "raw_attribute": "화면 밝기",
      "sentiment": "mixed"
    },
    {
      "raw_attribute": "HDMI 단자",
      "sentiment": "neutral"
    },
    {
      "raw_attribute": "받침대 내구성",
      "sentiment": "unknown"
    }
  ]
}

### Input

Product name:

{{product_name}}

Product category:

{{product_category}}

Review:

<review>
{{review}}
</review>

### Output
