import json
from typing import Dict, Any

def load_profile(profile_path: str) -> Dict[str, Any]:
    """Load the rich profile from JSON file."""
    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_family_section(family_data: Dict[str, Any]) -> str:
    """Generate the family section with detailed descriptions."""
    family_sections = []
    
    for relationship, member in family_data.items():
        if relationship == "mother":
            section = f"**{member['name']}**: Ashley's mother, described as {member['description'].lower()}."
        elif relationship == "father":
            section = f"**{member['name']}**: Ashley's father, {member['description'].lower()}."
        elif relationship == "older_brother":
            section = f"**{member['name']}**: Ashley's older brother, {member['description'].lower()}."
        elif relationship == "younger_sister":
            section = f"**{member['name']}**: Ashley's younger sister, {member['description'].lower()}."
        elif relationship == "grandfather":
            section = f"**{member['name']}**: Ashley's grandfather, {member['description'].lower()}."
        else:
            section = f"**{member['name']}**: {member['description']}."
        
        family_sections.append(section)
    
    return "\n".join(family_sections)

def generate_education_section(education_data: Dict[str, Any]) -> str:
    """Generate the education section."""
    school = education_data.get('high_school', {})
    return f"**{school.get('institution', 'School')}**: {school.get('description', 'Educational institution')}."

def generate_career_section(career_data: Dict[str, Any]) -> str:
    """Generate the career section."""
    position = career_data.get('current_position', {})
    return f"**{position.get('company', 'Company')}**: {position.get('description', 'Workplace description')}."

def generate_rich_response(profile_path: str) -> str:
    """Generate a rich, detailed response using the profile data."""
    profile = load_profile(profile_path)
    ashley_data = profile['ashley_profile']
    
    # Generate each section
    family_section = generate_family_section(ashley_data['family'])
    education_section = generate_education_section(ashley_data['education'])
    career_section = generate_career_section(ashley_data['career'])
    
    # Build the complete response
    response = f"""Here's a detailed profile of Ashley:

**Family:**
{family_section}

**Education:**
{education_section}

**Workplace:**
{career_section}

These details provide a comprehensive overview of Ashley's family, educational background, and professional life as a {ashley_data['career']['current_position']['title']}."""
    
    return response

def generate_alternative_response(profile_path: str) -> str:
    """Generate an alternative response format using response templates."""
    profile = load_profile(profile_path)
    ashley_data = profile['ashley_profile']
    
    # Extract key information for template filling
    family_members = ", ".join([f"{member['name']} ({member['relationship']})" 
                               for member in ashley_data['family'].values()])
    
    family_dynamic = "a strong support system with diverse personalities and roles"
    school_name = ashley_data['education']['high_school']['institution']
    school_description = ashley_data['education']['high_school']['description']
    job_title = ashley_data['career']['current_position']['title']
    company_name = ashley_data['career']['current_position']['company']
    company_description = ashley_data['career']['current_position']['description']
    career_summary = f"{job_title} in the {ashley_data['career']['current_position']['industry']} industry"
    
    # Use response templates
    response = f"""Ashley's Profile:

{ashley_data['response_templates']['family_section'].format(
    family_members=family_members, 
    family_dynamic=family_dynamic
)}

{ashley_data['response_templates']['education_section'].format(
    school_name=school_name, 
    school_description=school_description
)}

{ashley_data['response_templates']['career_section'].format(
    job_title=job_title, 
    company_name=company_name, 
    company_description=company_description
)}

{ashley_data['response_templates']['conclusion'].format(
    career_summary=career_summary
)}"""
    
    return response

if __name__ == "__main__":
    # Example usage
    profile_path = "data/documents/ashley_rich_profile.json"
    
    print("=== RICH RESPONSE GENERATION ===\n")
    print(generate_rich_response(profile_path))
    
    print("\n" + "="*50 + "\n")
    
    print("=== ALTERNATIVE TEMPLATE-BASED RESPONSE ===\n")
    print(generate_alternative_response(profile_path))
    
    print("\n" + "="*50 + "\n")
    
    # Show how to access specific data for custom queries
    profile = load_profile(profile_path)
    ashley_data = profile['ashley_profile']
    
    print("=== DATA ACCESS EXAMPLES ===\n")
    print(f"Ashley's strengths: {', '.join(ashley_data['personal_qualities']['strengths'])}")
    print(f"Family characteristics: {[member['characteristics'] for member in ashley_data['family'].values()]}")
    print(f"Company features: {', '.join(ashley_data['career']['current_position']['company_features'])}")
