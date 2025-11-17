class VoiceAgent:
    def __init__(self):
        self.memory = []
        self.context = {}
        self.tools = {}
        self.goals = []

    def perceive(self, audio_input: str) -> Dict[str, Any]:
        intent = self._extract_intent(audio_input)
        entities = self._extract_entities(audio_input)
        sentiment = self._analyze_sentiment(audio_input)
        perception = {
            'text': audio_input,
            'intent': intent,
            'entities': entities,
            'sentiment': sentiment,
            'timestamp': datetime.now().isoformat()
        }
        self.memory.append(perception)
        return perception

    def _extract_intent(self, text: str) -> str:
        text_lower = text.lower()
        intent_patterns = {
            'create': ['create', 'make', 'generate', 'write'],
            'search': ['search', 'find', 'look for', 'show me'],
            'analyze': ['analyze', 'explain', 'understand', 'what is'],
            'calculate': ['calculate', 'compute', 'how much', 'sum'],
            'schedule': ['schedule', 'plan', 'set reminder', 'meeting'],
            'translate': ['translate', 'say in', 'convert to'],
            'summarize': ['summarize', 'brief', 'tldr', 'overview']
        }
        for intent, keywords in intent_patterns.items():
            if any(kw in text_lower for kw in keywords):
                return intent
        return 'conversation'

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {
            'numbers': re.findall(r'\d+', text),
            'dates': re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text),
            'times': re.findall(r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b', text.lower()),
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        }
        return {k: v for k, v in entities.items() if v}

    def _analyze_sentiment(self, text: str) -> str:
        positive = ['good', 'great', 'excellent', 'happy', 'love', 'thank']
        negative = ['bad', 'terrible', 'sad', 'hate', 'angry', 'problem']
        text_lower = text.lower()
        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'

    # ✅ FIXED: moved inside class
    def reason(self, perception: Dict) -> Dict[str, Any]:
        intent = perception['intent']
        reasoning = {
            'goal': self._identify_goal(intent),
            'prerequisites': self._check_prerequisites(intent),
            'plan': self._create_plan(intent, perception['entities']),
            'confidence': self._calculate_confidence(perception)
        }
        return reasoning

    # ✅ FIXED: moved inside class
    def act(self, reasoning: Dict) -> str:
        plan = reasoning['plan']
        results = []
        for step in plan['steps']:
            result = self._execute_step(step)
            results.append(result)
        response = self._generate_response(results, reasoning)
        return response

    # ====== INTERNAL HELPERS (also moved inside class) ======

    def _identify_goal(self, intent: str) -> str:
        goal_mapping = {
            'create': 'Generate new content',
            'search': 'Retrieve information',
            'analyze': 'Understand and explain',
            'calculate': 'Perform computation',
            'schedule': 'Organize time-based tasks',
            'translate': 'Convert between languages',
            'summarize': 'Condense information'
        }
        return goal_mapping.get(intent, 'Assist user')

    def _check_prerequisites(self, intent: str) -> List[str]:
        prereqs = {
            'search': ['internet access', 'search tool'],
            'calculate': ['math processor'],
            'translate': ['translation model'],
            'schedule': ['calendar access']
        }
        return prereqs.get(intent, ['language understanding'])

    def _create_plan(self, intent: str, entities: Dict) -> Dict:
        plans = {
            'create': {'steps': ['understand_requirements', 'generate_content', 'validate_output'], 'estimated_time': '10s'},
            'analyze': {'steps': ['parse_input', 'analyze_components', 'synthesize_explanation'], 'estimated_time': '5s'},
            'calculate': {'steps': ['extract_numbers', 'determine_operation', 'compute_result'], 'estimated_time': '2s'}
        }
        default_plan = {'steps': ['understand_query', 'process_information', 'formulate_response'], 'estimated_time': '3s'}
        return plans.get(intent, default_plan)

    def _calculate_confidence(self, perception: Dict) -> float:
        base_confidence = 0.7
        if perception['entities']:
            base_confidence += 0.15
        if perception['sentiment'] != 'neutral':
            base_confidence += 0.1
        if len(perception['text'].split()) > 5:
            base_confidence += 0.05
        return min(base_confidence, 1.0)

    def _execute_step(self, step: str) -> Dict:
        return {'step': step, 'status': 'completed', 'output': f'Executed {step}'}

    def _generate_response(self, results: List, reasoning: Dict) -> str:
        intent = reasoning['goal']
        confidence = reasoning['confidence']
        prefix = "I understand you want to" if confidence > 0.8 else "I think you're asking me to"
        response = f"{prefix} {intent.lower()}. "
        if len(self.memory) > 1:
            response += "Based on our conversation, "
        response += f"I've analyzed your request and completed {len(results)} steps. "
        return response
