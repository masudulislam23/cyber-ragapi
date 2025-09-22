"""
LLM Utilities for profile categorization and other AI-powered features.
This module provides functions to interact with Large Language Models.
"""

import asyncio
import json
from typing import Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_llm_response(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Get a response from a local LLM using the specified prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        model: The model to use (default: gpt-3.5-turbo)
        
    Returns:
        The LLM response as a string
    """
    try:
        # Initialize local LLM (same as app.py)
        VLLM_API_BASE = "http://localhost:8002/v1"
        VLLM_MODEL_NAME = "neuralmagic/pixtral-12b-quantized.w4a16"
        
        local_llm = ChatOpenAI(
            temperature=0.1,  # Low temperature for consistent, structured responses
            model_name=VLLM_MODEL_NAME,
            openai_api_base=VLLM_API_BASE,
            openai_api_key="token-abc123",  # vLLM often doesn't require a real key for internal setups
            max_tokens=1000
        )
        
        # Use the same pattern as app.py
        response = await local_llm.ainvoke([HumanMessage(content=prompt)])
        
        # Extract JSON content from markdown response if present
        content = response.content.strip()
        
        # Check if response is wrapped in markdown code blocks
        if content.startswith("```json") and content.endswith("```"):
            # Extract content between ```json and ```
            json_start = content.find("```json") + 7
            json_end = content.rfind("```")
            if json_end > json_start:
                content = content[json_start:json_end].strip()
        elif content.startswith("```") and content.endswith("```"):
            # Extract content between ``` and ``` (any language)
            json_start = content.find("```") + 3
            json_end = content.rfind("```")
            if json_end > json_start:
                content = content[json_start:json_end].strip()
        
        return content
            
    except Exception as e:
        logger.error(f"Error getting local LLM response: {str(e)}")
        logger.info("Falling back to mock response")
        # Fallback to mock response
        return await get_mock_llm_response(prompt)

async def get_mock_llm_response(prompt: str) -> str:
    """
    Provide a mock LLM response for testing when OpenAI is not available.
    This simulates intelligent categorization based on common patterns.
    """
    try:
        print(f"Mock LLM processing prompt: {prompt[:200]}...")
        
        # Simple pattern matching for common profile information
        prompt_lower = prompt.lower()
        
        # Extract person name - try multiple patterns
        person_name = "Unknown"
        name_patterns = [
            "**person**: ",
            "person: ",
            "person**: ",
            "**person**:"
        ]
        for pattern in name_patterns:
            if pattern in prompt:
                person_name = prompt.split(pattern)[1].split("\n")[0].strip()
                break
        
        print(f"Extracted person name: {person_name}")
        
        # Extract information to categorize - try multiple patterns
        info_lines = []
        info_patterns = [
            "**information to categorize**:",
            "**information to analyze**:",
            "information to categorize:",
            "information to analyze:"
        ]
        
        for pattern in info_patterns:
            if pattern in prompt_lower:
                info_start = prompt.find(pattern)
                if info_start != -1:
                    info_section = prompt[info_start:]
                    # Look for the next section marker
                    next_section_markers = [
                        "**extract and categorize",
                        "**extraction rules",
                        "**return only",
                        "**examples"
                    ]
                    info_end = len(info_section)
                    for marker in next_section_markers:
                        marker_pos = info_section.find(marker)
                        if marker_pos != -1 and marker_pos < info_end:
                            info_end = marker_pos
                    
                    info_text = info_section[:info_end].strip()
                    # Extract bullet points
                    info_lines = [line.strip("- ").strip() for line in info_text.split("\n") if line.strip().startswith("-")]
                    break
        
        print(f"Extracted info lines: {info_lines}")
        
        if not info_lines:
            print("No information lines found, trying alternative parsing...")
            # Fallback: look for any lines starting with "-" in the entire prompt
            info_lines = [line.strip("- ").strip() for line in prompt.split("\n") if line.strip().startswith("-")]
            print(f"Fallback info lines: {info_lines}")
        
        # Mock categorization logic
        result = {
            "location": None,
            "phone": None,
            "education": None,
            "workplace": None,
            "family": [],
            "other": []
        }
        
        for info in info_lines:
            if not info:  # Skip empty lines
                continue
                
            info_lower = info.lower()
            print(f"Processing info: '{info}'")
            
            # Location detection - extract just the place name
            if any(word in info_lower for word in ['lives in', 'lived in', 'from', 'in iceland', 'in europe', 'going to', 'traveling to']):
                if 'iceland' in info_lower:
                    result["location"] = "Iceland"
                elif 'europe' in info_lower:
                    result["location"] = "Europe"
                elif 'new york' in info_lower:
                    result["location"] = "New York"
                elif 'california' in info_lower:
                    result["location"] = "California"
                elif 'london' in info_lower:
                    result["location"] = "London"
                elif 'chicago' in info_lower:
                    result["location"] = "Chicago"
                elif 'boston' in info_lower:
                    result["location"] = "Boston"
                elif 'miami' in info_lower:
                    result["location"] = "Miami"
                elif 'seattle' in info_lower:
                    result["location"] = "Seattle"
                else:
                    # Try to extract other locations intelligently
                    import re
                    # Look for patterns like "in [Location]" or "from [Location]" or "going to [Location]"
                    location_patterns = [
                        r'in\s+([a-zA-Z\s]+?)(?:\s|,|and|$|\.)',
                        r'from\s+([a-zA-Z\s]+?)(?:\s|,|and|$|\.)',
                        r'lives\s+in\s+([a-zA-Z\s]+?)(?:\s|,|and|$|\.)',
                        r'going\s+to\s+([a-zA-Z\s]+?)(?:\s|,|and|$|\.)',
                        r'traveling\s+to\s+([a-zA-Z\s]+?)(?:\s|,|and|$|\.)'
                    ]
                    for pattern in location_patterns:
                        match = re.search(pattern, info_lower)
                        if match:
                            location = match.group(1).strip().title()
                            if len(location) > 2:  # Avoid very short matches
                                result["location"] = location
                                print(f"Extracted location: {location}")
                                break
                
                # If we found a location, continue to next info
                if result["location"]:
                    continue
            
            # Phone detection - extract just the number
            elif any(word in info_lower for word in ['phone', 'number', 'contact']):
                import re
                # Look for various phone number formats
                number_patterns = [
                    r'(\d{5,})',  # Basic 5+ digit number
                    r'(\d{3}-\d{3}-\d{4})',  # XXX-XXX-XXXX format
                    r'(\d{3}\.\d{3}\.\d{4})',  # XXX.XXX.XXXX format
                    r'(\d{3}\s\d{3}\s\d{4})'   # XXX XXX XXXX format
                ]
                for pattern in number_patterns:
                    number_match = re.search(pattern, info)
                    if number_match:
                        result["phone"] = number_match.group(1)
                        print(f"Extracted phone: {result['phone']}")
                        break
                continue
            
            # Education detection - keep full context
            elif any(word in info_lower for word in ['school', 'university', 'college', 'academy', 'studied', 'st.', 'graduated', 'degree']):
                result["education"] = info
                print(f"Extracted education: {info}")
                continue
            
            # Workplace detection - keep full context
            elif any(word in info_lower for word in ['work', 'job', 'company', 'engineer', 'developer', 'works at', 'works for', 'employed', 'career']):
                result["workplace"] = info
                print(f"Extracted workplace: {info}")
                continue
            
            # Family detection - add to array
            elif any(word in info_lower for word in ['sister', 'brother', 'mother', 'father', 'parent', 'family', 'sibling']):
                result["family"].append(info)
                print(f"Extracted family info: {info}")
                continue
            
            # Other information - add to array
            else:
                result["other"].append(info)
                print(f"Added to other: {info}")
        
        # Clean up empty lists
        if not result["family"]:
            result["family"] = None
        if not result["other"]:
            result["other"] = None
        
        print(f"Final mock categorization result: {result}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        print(f"Error in mock LLM response: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # Return a safe default response
        return json.dumps({
            "location": None,
            "phone": None,
            "education": None,
            "workplace": None,
            "family": None,
            "other": None
        })

async def test_llm_categorization():
    """Test the LLM categorization functionality."""
    print("Testing LLM Profile Categorization")
    print("=" * 60)
    
    # Test prompt
    test_prompt = """
You are a profile information categorizer. Your task is to analyze information about a person and categorize it into clean, structured fields.

**Person**: Marry

**Information to categorize**:
- going to Europe and phone number is 232043
- lives in iceland
- has two sisters named Isanur and Elizabeth
- her number is 5555554

**Categorize this information into the following fields**:
- location: Where the person lives, is from, or is located
- phone: Phone number, contact number, or phone information
- education: Schools, universities, studies, academic background
- workplace: Job, company, work, profession, career
- family: Family members, relatives, siblings, parents
- other: Any other relevant information that doesn't fit the above categories

**Rules**:
1. Extract only the essential information for each field
2. For location: Extract just the place name (e.g., "Iceland", "New York")
3. For phone: Extract just the number (e.g., "5555554")
4. For education: Keep the full educational information
5. For workplace: Keep the full work information
6. For family: Keep the full family information
7. For other: Put anything that doesn't clearly fit above categories

**Return ONLY a valid JSON object** with this exact structure:
{
    "location": "extracted_location_or_null",
    "phone": "extracted_phone_or_null",
    "education": "extracted_education_or_null",
    "workplace": "extracted_workplace_or_null",
    "family": ["family_info1", "family_info2"] or null,
    "other": ["other_info1", "other_info2"] or null
}

**Example**:
Input: "lives in iceland, has two sisters named Isanur and Elizabeth, her number is 5555554"
Output: {
    "location": "Iceland",
    "phone": "5555554",
    "family": ["has two sisters named Isanur and Elizabeth"],
    "other": null
}

Now categorize the information for Marry:
"""
    
    print("Test Prompt:")
    print(test_prompt)
    
    print("\nGetting LLM Response...")
    
    try:
        response = await get_llm_response(test_prompt)
        print("LLM Response Received:")
        print(response)
        
        # Try to parse as JSON
        try:
            parsed = json.loads(response)
            print("\nParsed JSON:")
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError as e:
            print(f"\nFailed to parse JSON: {str(e)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_llm_categorization())
