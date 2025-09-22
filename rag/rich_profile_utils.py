"""
Rich Profile Utilities for enhanced profile generation and response formatting.
This module provides functions to generate rich, detailed responses from profile data.
"""

import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import re

def generate_rich_family_section(family_data: Dict[str, Any], person_name: str = "Unknown") -> str:
    """Generate a rich family section with detailed descriptions."""
    if not family_data:
        return "No family information available."
    
    family_sections = []
    
    for relationship, member in family_data.items():
        if not member or not isinstance(member, dict):
            continue
            
        name = member.get('name', 'Unknown')
        description = member.get('description', '')
        characteristics = member.get('characteristics', [])
        personality_traits = member.get('personality_traits', '')
        
        # Build the section
        section_parts = [f"**{name}**"]
        
        if relationship == "mother":
            section_parts.append(f"{person_name}'s mother")
        elif relationship == "father":
            section_parts.append(f"{person_name}'s father")
        elif relationship == "older_brother":
            section_parts.append(f"{person_name}'s older brother")
        elif relationship == "younger_sister":
            section_parts.append(f"{person_name}'s younger sister")
        elif relationship == "grandfather":
            section_parts.append(f"{person_name}'s grandfather")
        elif relationship == "grandmother":
            section_parts.append(f"{person_name}'s grandmother")
        else:
            section_parts.append(f"{person_name}'s {relationship.replace('_', ' ')}")
        
        if description:
            section_parts.append(f"described as {description.lower()}")
        
        if characteristics:
            char_text = ", ".join(characteristics)
            section_parts.append(f"with characteristics: {char_text}")
        
        if personality_traits:
            section_parts.append(f"Personality: {personality_traits}")
        
        section = ": ".join(section_parts) + "."
        family_sections.append(section)
    
    return "\n".join(family_sections) if family_sections else "No family information available."

def generate_rich_education_section(education_data: Dict[str, Any]) -> str:
    """Generate a rich education section."""
    if not education_data:
        return "No education information available."
    
    education_sections = []
    
    for level, info in education_data.items():
        if not info or not isinstance(info, dict):
            continue
            
        institution = info.get('institution', 'Unknown School')
        description = info.get('description', '')
        features = info.get('features', [])
        reputation = info.get('reputation', '')
        
        section_parts = [f"**{institution}**"]
        
        if description:
            section_parts.append(description)
        
        if features:
            features_text = ", ".join(features)
            section_parts.append(f"Features: {features_text}")
        
        if reputation:
            section_parts.append(f"Reputation: {reputation}")
        
        section = ": ".join(section_parts) + "."
        education_sections.append(section)
    
    return "\n".join(education_sections) if education_sections else "No education information available."

def generate_rich_career_section(career_data: Dict[str, Any]) -> str:
    """Generate a rich career section."""
    if not career_data:
        return "No career information available."
    
    career_sections = []
    
    for position_type, info in career_data.items():
        if not info or not isinstance(info, dict):
            continue
            
        title = info.get('title', 'Unknown Position')
        company = info.get('company', 'Unknown Company')
        description = info.get('description', '')
        industry = info.get('industry', '')
        company_features = info.get('company_features', [])
        
        section_parts = [f"**{company}**"]
        
        if title:
            section_parts.append(f"where Ashley works as a {title}")
        
        if description:
            section_parts.append(description)
        
        if industry:
            section_parts.append(f"Industry: {industry}")
        
        if company_features:
            features_text = ", ".join(company_features)
            section_parts.append(f"Company features: {features_text}")
        
        section = ": ".join(section_parts) + "."
        career_sections.append(section)
    
    return "\n".join(career_sections) if career_sections else "No career information available."

def generate_rich_response_from_profile(profile_data: Dict[str, Any]) -> str:
    """Generate a rich, detailed response from profile data."""
    if not profile_data:
        return "I'm sorry, but I don't have any profile information available at the moment."
    
    # Get the person's name from the profile
    person_name = profile_data.get('name', 'Unknown')
    
    # Get relationship
    relationship = profile_data.get('relationship', 'friend')
    
    # Build the complete response with a warm, professional tone
    response_parts = [f"Of course! I'd be happy to share what I know about {person_name}."]
    
    # Add a friendly introduction based on relationship
    if relationship.lower() in ['friend', 'colleague', 'family']:
        response_parts.append(f"\n{person_name} is your {relationship}, and here are the details I have:")
    else:
        response_parts.append(f"\nHere are the details I have about {person_name}:")
    
    # Add location if available
    location = profile_data.get('location', '')
    if location:
        response_parts.append(f"\n**Location**: {person_name} lives in {location}")
    
    # Add phone number if available
    phone = profile_data.get('phone', '')
    if phone:
        response_parts.append(f"\n**Phone**: {person_name}'s phone number is {phone}")
    
    # Add birthday if available
    birthday = profile_data.get('birthday', '')
    if birthday:
        response_parts.append(f"\n**Birthday**: {person_name}'s birthday is {birthday}")
    
    # Add education if available
    education = profile_data.get('education', '')
    if education:
        response_parts.append(f"\n**Education**: {education}")
    
    # Add workplace if available
    workplace = profile_data.get('workplace', '')
    if workplace:
        response_parts.append(f"\n**Workplace**: {workplace}")
    
    # Add family information if available
    family = profile_data.get('family', [])
    if family:
        response_parts.append(f"\n**Family**:")
        for member in family:
            response_parts.append(f"- {member}")
    
    # Add other information if available
    other = profile_data.get('other', [])
    if other:
        response_parts.append(f"\n**Additional Information**:")
        for info in other:
            response_parts.append(f" - {info}")
    
    # Add a friendly closing
    response_parts.append(f"\nThat's everything I know about {person_name}! If you'd like to add or update any information, just let me know.")
    
    return "\n".join(response_parts)

def is_rich_profile(profile_data: Dict[str, Any]) -> bool:
    """Check if a profile is a clean, usable profile (has structured information)."""
    if not profile_data or not isinstance(profile_data, dict):
        return False
    
    # Check for clean profile indicators
    clean_indicators = ['location', 'phone', 'birthday', 'education', 'workplace', 'family', 'other']
    has_categorized_info = any(indicator in profile_data for indicator in clean_indicators)
    
    # Check if profile still has raw information field (indicates it's not fully converted)
    has_raw_information = 'information' in profile_data and profile_data['information']
    
    # A profile is considered rich if it has categorized information AND no raw information field
    # OR if it has complete basic structure without raw information
    has_basic_info = all(key in profile_data for key in ['name', 'relationship', 'created_date', 'last_updated'])
    
    # Profile is rich if it has categorized info AND no raw info, OR basic structure without raw info
    return (has_categorized_info and not has_raw_information) or (has_basic_info and not has_raw_information)

async def convert_simple_to_rich_profile(simple_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a simple profile to a clean, usable profile structure using LLM for intelligent field detection."""
    if not simple_profile or not isinstance(simple_profile, dict):
        print(f"Invalid simple profile: {simple_profile}")
        return {}
    
    name = simple_profile.get('name', 'Unknown')
    relationship = simple_profile.get('relationship', 'unknown')
    information = simple_profile.get('information', [])
    
    print(f"Converting profile for {name} with information: {information}")
    
    # Filter out meta-information before processing
    # Handle both string and list formats for information
    if isinstance(information, str):
        # If information is a string, split it into parts for filtering
        information_parts = [part.strip() for part in information.split(' and ') if part.strip()]
        filtered_information = filter_meta_information(information_parts)
        # Rejoin the filtered parts
        filtered_information = ' and '.join(filtered_information) if filtered_information else ""
    else:
        # If information is already a list, filter it directly
        filtered_information = filter_meta_information(information)
    
    if filtered_information != information:
        print(f"Filtered out meta-information for {name}. Original: {information}, Filtered: {filtered_information}")
    
    # Create clean, usable profile structure
    # Start with existing structured fields if they exist
    clean_profile = {
        "name": name,
        "relationship": relationship,
        "created_date": simple_profile.get('timestamp', simple_profile.get('created_date', datetime.now().isoformat())),
        "last_updated": datetime.now().isoformat()
    }
    
    # Preserve existing structured fields (phone, location, etc.) if they exist
    existing_structured_fields = ['phone', 'location', 'birthday', 'education', 'workplace', 'family', 'other']
    for field in existing_structured_fields:
        if field in simple_profile and simple_profile[field]:
            clean_profile[field] = simple_profile[field]
    
    if not filtered_information:
        print(f"No valid information to categorize for {name}, returning basic profile")
        return clean_profile
    
    # Use LLM to intelligently categorize information
    try:
        print(f"Attempting LLM categorization for {name} with filtered info: {filtered_information}")
        categorized_info = await categorize_information_with_llm(name, filtered_information)
        print(f"LLM categorization successful for {name}: {categorized_info}")
        
        # Add categorized information to profile
        for field, value in categorized_info.items():
            if value:  # Only add non-empty fields
                clean_profile[field] = value
                
    except Exception as e:
        print(f"LLM categorization failed for {name}, falling back to fallback LLM: {str(e)}")
        # Fallback to secondary LLM categorization
        try:
            print(f"Attempting fallback LLM categorization for {name}")
            categorized_info = await categorize_information_with_keywords(filtered_information)
            print(f"Fallback LLM categorization result for {name}: {categorized_info}")
            
            for field, value in categorized_info.items():
                if value:
                    clean_profile[field] = value
        except Exception as fallback_error:
            print(f"Fallback LLM also failed for {name}: {str(fallback_error)}")
            # Ultimate fallback - add filtered information to 'other' field to ensure profile is not empty
            if filtered_information:
                clean_profile['other'] = filtered_information
                print(f"Added filtered information to 'other' field for {name}: {filtered_information}")
    
    print(f"Final converted profile for {name}: {clean_profile}")
    return clean_profile

async def categorize_information_with_keywords(information: Union[str, List[str]]) -> Dict[str, Any]:
    """Fallback LLM-based categorization when main LLM fails."""
    try:
        # Use a simpler LLM prompt for fallback
        if isinstance(information, str):
            # If information is a string, split it into parts
            info_parts = [part.strip() for part in information.split(' and ') if part.strip()]
            info_text = "\n".join([f"- {info}" for info in info_parts])
        else:
            # If information is already a list
            info_text = "\n".join([f"- {info}" for info in information])
        
        fallback_prompt = f"""
Analyze this information and extract key details:

{info_text}

Extract and return ONLY a JSON object with these fields:
- location: Just the place name (e.g., "Iceland", "New York")
- phone: Just the phone number (e.g., "5555554")
- birthday: Just the birthday (e.g., "January 15", "March 22nd")
- education: Full educational information
- workplace: Full work information  
- family: Array of family information
- other: Array of other information

Return valid JSON only:
"""
        
        from rag.llm_utils import get_llm_response
        response = await get_llm_response(fallback_prompt)
        
        try:
            import json
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # If even fallback LLM fails, return empty structure
            return {
                "location": None,
                "phone": None,
                "birthday": None,
                "education": None,
                "workplace": None,
                "family": None,
                "other": None
            }
            
    except Exception as e:
        print(f"Fallback LLM also failed: {str(e)}")
        # Ultimate fallback - return empty structure
        return {
            "location": None,
            "phone": None,
            "birthday": None,
            "education": None,
            "workplace": None,
            "family": None,
            "other": None
        }

async def categorize_information_with_llm(person_name: str, information: Union[str, List[str]]) -> Dict[str, Any]:
    """Use LLM to intelligently categorize profile information into clean fields."""
    try:
        from rag.llm_utils import get_llm_response
        
        # Prepare the information for LLM analysis
        if isinstance(information, str):
            # If information is a string, split it into parts
            info_parts = [part.strip() for part in information.split(' and ') if part.strip()]
            info_text = "\n".join([f"- {info}" for info in info_parts])
        else:
            # If information is already a list
            info_text = "\n".join([f"- {info}" for info in information])
        
        prompt = f"""
        You are an expert profile information extractor and categorizer. Your task is to intelligently analyze information about a person and extract specific details into clean, structured fields.

        **Person**: {person_name}

        **Information to categorize**:
        {info_text}

        **Extract and categorize this information into the following fields**:

        **location**: Extract ONLY the place name where the person lives, is from, or is located
        - Examples: "Iceland", "New York", "London", "California"
        - Extract just the location name, not the full sentence
        - If multiple locations mentioned, choose the primary one

        **phone**: Extract ONLY the phone number or contact number
        - Examples: "5555554", "123-456-7890", "5551234"
        - Extract just the number, not the full sentence
        - If multiple numbers, choose the most recent/primary one

        **birthday**: Extract ONLY the birthday or birth date
        - Examples: "January 15", "March 22nd", "December 3", "15th of January"
        - Extract just the date, not the full sentence
        - If multiple dates mentioned, choose the most likely birthday

        **education**: Extract the full educational information
        - Examples: "studied at St. Catherine's Academy", "graduated from Harvard University"
        - Keep the complete educational context

        **workplace**: Extract the full work/professional information
        - Examples: "works as software engineer at Horizon Tech Innovations", "employed at Google"
        - Keep the complete work context

                 **family**: Extract family-related information as an array
         - Examples: ["has two sisters: Isanur and Elizabeth", "mother is a doctor", "has a brother named John"]
         - Use natural, grammatically correct descriptions
         - Each family fact as a separate array item

        **other**: Extract any other relevant information that doesn't fit above categories
        - Examples: ["going to Europe", "likes basketball", "has a dog named Max"]
        - Each fact as a separate array item

                 **Extraction Rules**:
         1. **Be precise**: Extract only the essential information for each field
         2. **Location**: Just the place name, not "lives in" or "from"
         3. **Phone**: Just the number, not "phone number is" or "contact"
         4. **Education**: Keep the full educational context
         5. **Workplace**: Keep the full work context
         6. **Family**: Array of family facts with natural, grammatically correct descriptions
         7. **Other**: Array of other relevant facts

        **Return ONLY a valid JSON object** with this exact structure:
        {{
            "location": "extracted_location_or_null",
            "phone": "extracted_phone_or_null",
            "birthday": "extracted_birthday_or_null",
            "education": "extracted_education_or_null",
            "workplace": "extracted_workplace_or_null",
            "family": ["family_fact1", "family_fact2"] or null,
            "other": ["other_fact1", "other_fact2"] or null
        }}

        **Examples**:

        Input: "lives in iceland, has two sisters named Isanur and Elizabeth, her number is 5555554, birthday is January 15"
         Output: {{
             "location": "Iceland",
             "phone": "5555554",
             "birthday": "January 15",
             "family": ["has two sisters: Isanur and Elizabeth"],
             "other": null
         }}

        Input: "studied at St. Catherine's Academy, works as software engineer at Horizon Tech Innovations, birthday is March 22nd"
        Output: {{
            "location": null,
            "phone": null,
            "birthday": "March 22nd",
            "education": "studied at St. Catherine's Academy",
            "workplace": "works as software engineer at Horizon Tech Innovations",
            "family": null,
            "other": null
        }}

        Now extract and categorize the information for {person_name}:
        """
        
        # Get LLM response
        response = await get_llm_response(prompt)
        
        # Parse the JSON response
        try:
            import json
            categorized_data = json.loads(response.strip())
            
            # Validate the structure
            expected_fields = ['location', 'phone', 'education', 'workplace', 'family', 'other']
            for field in expected_fields:
                if field not in categorized_data:
                    categorized_data[field] = None
            
            # Clean up the data (remove None values, ensure lists for family/other)
            cleaned_data = {}
            for field, value in categorized_data.items():
                if value is not None:
                    if field in ['family', 'other'] and not isinstance(value, list):
                        cleaned_data[field] = [value] if value else []
                    else:
                        cleaned_data[field] = value
            
            return cleaned_data
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {str(e)}")
            print(f"LLM Response: {response}")
            raise Exception("LLM response is not valid JSON")
            
    except Exception as e:
        print(f"Error in LLM categorization: {str(e)}")
        raise e

async def enhance_existing_profile(profile_data: Dict[str, Any], additional_info: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance an existing profile with additional rich information."""
    if not profile_data:
        return additional_info
    
    # If it's already a rich profile, merge the additional info
    if is_rich_profile(profile_data):
        enhanced_profile = profile_data.copy()
        
        # Merge additional information
        for key, value in additional_info.items():
            if key in enhanced_profile and isinstance(enhanced_profile[key], dict) and isinstance(value, dict):
                enhanced_profile[key].update(value)
            else:
                enhanced_profile[key] = value
        
        enhanced_profile["last_updated"] = datetime.now().isoformat()
        return enhanced_profile
    
    # If it's a simple profile, convert it first
    rich_profile = await convert_simple_to_rich_profile(profile_data)
    return await enhance_existing_profile(rich_profile, additional_info)

async def convert_all_profiles_to_rich(user_id: str) -> bool:
    """
    Convert all existing simple profiles for a user to rich format.
    
    Args:
        user_id: ID of the user
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from rag.database import load_user_profiles, save_user_profiles
        
        # Load existing profiles
        profiles = await load_user_profiles(user_id)
        
        if not profiles:
            return True  # No profiles to convert
        
        # Convert each profile to rich format
        converted_count = 0
        for name_lower, profile in profiles.items():
            if not is_rich_profile(profile):
                try:
                    rich_profile = await convert_simple_to_rich_profile(profile)
                    profiles[name_lower] = rich_profile
                    converted_count += 1
                    print(f"Converted profile for {profile.get('name', 'Unknown')} to rich format")
                except Exception as e:
                    print(f"Failed to convert profile for {profile.get('name', 'Unknown')}: {str(e)}")
                    continue
        
        if converted_count > 0:
            # Save the converted profiles
            success = await save_user_profiles(user_id, profiles)
            if success:
                print(f"Successfully converted {converted_count} profiles to rich format for user {user_id}")
                return True
            else:
                print(f"Failed to save converted profiles for user {user_id}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error converting profiles to rich format for user {user_id}: {str(e)}")
        return False

async def intelligently_update_profile(existing_profile: Dict[str, Any], new_information: List[str]) -> Dict[str, Any]:
    """
    Intelligently update an existing profile by merging new information while preserving existing data.
    
    Args:
        existing_profile: The current profile data
        new_information: List of new information strings to process
        
    Returns:
        Updated profile with merged information
    """
    if not existing_profile or not isinstance(existing_profile, dict):
        print(f"Invalid existing profile: {existing_profile}")
        return {}
    
    # Create a copy to avoid modifying the original
    updated_profile = existing_profile.copy()
    
    # Always update the last_updated timestamp
    updated_profile['last_updated'] = datetime.now().isoformat()
    
    # Never modify created_date - it should remain fixed
    
    # Filter out meta-information before processing
    filtered_information = filter_meta_information(new_information)
    if filtered_information != new_information:
        print(f"Filtered out meta-information. Original: {new_information}, Filtered: {filtered_information}")
    
    # Process each piece of filtered information
    fields_needing_confirmation = []
    
    for info in filtered_information:
        if not info:
            continue
            
        print(f"Processing new information: {info}")
        
        # Use LLM to categorize this new information
        try:
            categorized_info = await categorize_information_with_llm(
                updated_profile.get('name', 'Unknown'), 
                [info]
            )
            print(f"Categorized new info: {categorized_info}")
            
            # Merge the categorized information intelligently
            for field, value in categorized_info.items():
                if not value:
                    continue
                    
                if field in ['family', 'other']:
                    # For array fields, append new information
                    if field not in updated_profile:
                        updated_profile[field] = []
                    elif not isinstance(updated_profile[field], list):
                        updated_profile[field] = []
                    
                    # Add new items if they don't already exist
                    if isinstance(value, list):
                        for item in value:
                            if item not in updated_profile[field]:
                                updated_profile[field].append(item)
                                print(f"Added to {field}: {item}")
                    else:
                        if value not in updated_profile[field]:
                            updated_profile[field].append(value)
                            print(f"Added to {field}: {value}")
                            
                elif field in ['location', 'phone', 'birthday', 'education', 'workplace']:
                    # For single-value fields, check if they're different
                    current_value = updated_profile.get(field)
                    if current_value != value:
                        print(f"Field '{field}' has different value:")
                        print(f"  Current: {current_value}")
                        print(f"  New: {value}")
                        
                        # Update the profile with the new value immediately
                        updated_profile[field] = value
                        print(f"Updated {field} to: {value}")
                        
                        # Collect fields that need confirmation for user notification
                        fields_needing_confirmation.append({
                            'field': field,
                            'current_value': current_value,
                            'new_value': value
                        })
                        
        except Exception as e:
            print(f"Failed to categorize new information '{info}': {str(e)}")
            # Fallback: add to 'other' field
            if 'other' not in updated_profile:
                updated_profile['other'] = []
            elif not isinstance(updated_profile['other'], list):
                updated_profile['other'] = []
            
            if info not in updated_profile['other']:
                updated_profile['other'].append(info)
                print(f"Added uncategorized info to 'other': {info}")
    
    # If there are fields needing confirmation, ask about them all at once
    if fields_needing_confirmation:
        # Create a comprehensive confirmation message for all fields
        additional_info = []
        if updated_profile.get('other'):
            additional_info.extend(updated_profile['other'])
        if updated_profile.get('family'):
            additional_info.extend(updated_profile['family'])
        
        # Build confirmation message for all fields that need confirmation
        confirmation_details = []
        for field_info in fields_needing_confirmation:
            field_name = field_info['field']
            current_val = field_info['current_value']
            new_val = field_info['new_value']
            confirmation_details.append(f"  {field_name}: {current_val or 'Not set'} -> {new_val}")
        
        # Add context about additional information
        context_message = ""
        if additional_info:
            context_message = f"\n\nI'll also add this additional information: {', '.join(additional_info)}"
        
        # Ask for confirmation on all fields at once
        await confirm_profile_update_multi(
            fields_needing_confirmation,
            updated_profile.get('name', 'Unknown'),
            confirmation_details,
            context_message,
            updated_profile
        )
    
    print(f"Final updated profile: {updated_profile}")
    return updated_profile

async def confirm_profile_update_multi(fields_needing_confirmation: List[Dict[str, Any]], person_name: str, confirmation_details: List[str], context_message: str = "", updated_profile: Dict[str, Any] = None) -> bool:
    """
    Ask user for confirmation before updating multiple critical profile fields.
    
    Args:
        fields_needing_confirmation: List of fields that need confirmation
        person_name: Name of the person whose profile is being updated
        confirmation_details: List of formatted strings showing field changes
        context_message: Additional context about what else will be added
        updated_profile: The complete updated profile that should be saved if confirmed
        
    Returns:
        True if user confirms, False otherwise
    """
    # Build the confirmation message for multiple fields
    confirmation_message = f"""
I notice you want to update {person_name}'s profile with the following changes:

{chr(10).join(confirmation_details)}{context_message}

Would you like me to update all this information? (yes/no)
"""
    
    print(confirmation_message)
    
    # Raise a special exception to pause the update and request confirmation
    raise ProfileUpdateConfirmationRequired(
        field_name="multiple_fields",  # Special indicator for multiple fields
        current_value=None,
        new_value=None,
        person_name=person_name,
        confirmation_message=confirmation_message,
        updated_profile=updated_profile,
        fields_needing_confirmation=fields_needing_confirmation  # Store all fields that need confirmation
    )

async def confirm_profile_update(field_name: str, current_value: Any, new_value: Any, person_name: str, context_message: str = "", updated_profile: Dict[str, Any] = None) -> bool:
    """
    Ask user for confirmation before updating a critical profile field.
    
    Args:
        field_name: Name of the field being updated
        current_value: Current value in the profile
        new_value: New value to be set
        person_name: Name of the person whose profile is being updated
        context_message: Additional context about what else will be added
        updated_profile: The complete updated profile that should be saved if confirmed
        
    Returns:
        True if user confirms, False otherwise
    """
    # This function now raises a special exception to pause the update process
    # and wait for user confirmation in the chat flow
    
    confirmation_message = f"""
    I notice you want to update {person_name}'s {field_name}:
    Current: {current_value or 'Not set'}
    New: {new_value}{context_message}

    Would you like me to update this information? (yes/no)
    """
    
    print(confirmation_message)
    
    # Raise a special exception to pause the update and request confirmation
    raise ProfileUpdateConfirmationRequired(
        field_name=field_name,
        current_value=current_value,
        new_value=new_value,
        person_name=person_name,
        confirmation_message=confirmation_message,
        updated_profile=updated_profile,
        fields_needing_confirmation=[]  # Empty list for single field updates
    )

class ProfileUpdateConfirmationRequired(Exception):
    """Exception raised when profile update requires user confirmation."""
    
    def __init__(self, field_name: str, current_value: Any, new_value: Any, person_name: str, confirmation_message: str, updated_profile: Dict[str, Any] = None, fields_needing_confirmation: List[Dict[str, Any]] = None):
        self.field_name = field_name
        self.current_value = current_value
        self.new_value = new_value
        self.person_name = person_name
        self.confirmation_message = confirmation_message
        self.updated_profile = updated_profile  # Store the complete updated profile
        self.fields_needing_confirmation = fields_needing_confirmation or []  # Store all fields that need confirmation
        super().__init__(confirmation_message)

def filter_meta_information(information: List[str]) -> List[str]:
    """
    Filter out meta-information that shouldn't be stored in profiles.
    
    Args:
        information: List of information strings to filter
        
    Returns:
        Filtered list with meta-information removed
    """
    if not information:
        return []
    
    # Patterns that indicate meta-information (requests, questions, etc.)
    meta_patterns = [
        r'asking\s+to\s+create\s+a\s+profile',
        r'asking\s+about\s+more\s+info',
        r'asking\s+for\s+.*profile',
        r'requesting\s+.*profile',
        r'want\s+to\s+create\s+.*profile',
        r'need\s+to\s+create\s+.*profile',
        r'would\s+like\s+to\s+create\s+.*profile',
        r'asking\s+.*relationship',
        r'asking\s+.*name',
        r'asking\s+.*information',
        r'requesting\s+.*information',
        r'want\s+.*information',
        r'need\s+.*information'
    ]
    
    filtered_info = []
    for info in information:
        # Check if this information matches any meta-patterns
        is_meta = False
        for pattern in meta_patterns:
            if re.search(pattern, info.lower()):
                is_meta = True
                break
        
        # Also check for common meta-phrases
        meta_phrases = [
            'asking to create',
            'asking about more info',
            'asking for profile',
            'requesting profile',
            'want to create',
            'need to create',
            'would like to create',
            'asking relationship',
            'asking name',
            'asking information',
            'requesting information',
            'want information',
            'need information'
        ]
        
        if not is_meta:
            for phrase in meta_phrases:
                if phrase.lower() in info.lower():
                    is_meta = True
                    break
        
        # Only add non-meta information
        if not is_meta:
            filtered_info.append(info)
    
    return filtered_info
