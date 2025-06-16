import pandas as pd
import streamlit as st
import csv
import io
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import uuid

# No longer need pyecore, so PYECORE_AVAILABLE check is removed.
# The app can now reliably generate XMI without this external dependency.

# Attempt to import SATELLITE_OPTIONS for model generation
try:
    from constellation_sim import SATELLITE_OPTIONS
except ImportError:
    # This will be checked in generate_xmi_model and a warning will be shown.
    SATELLITE_OPTIONS = None

def get_int_param_from_series(series, key, default_val=0):
    """Safely retrieves and converts a parameter to int from a pandas Series."""
    val = series.get(key)
    if pd.isna(val) or val is None:
        return default_val
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default_val

def generate_requirements_csv(optimal_design: pd.Series, constraints: dict) -> str:
    """
    Generates a CSV string for import into requirements management tools like DOORS or JAMA.
    (This function remains unchanged as it is independent of XMI generation).
    """
    if optimal_design is None or optimal_design.empty:
        st.warning("No optimal design selected. Cannot generate requirements CSV.")
        return None
        
    output = io.StringIO()
    writer = csv.writer(output)
    header = ["ID", "Name", "Description", "Type", "Depends On"]
    writer.writerow(header)

    writer.writerow(["KPP-001", "Max Cost", f"The total system lifecycle cost shall not exceed ${constraints.get('max_cost_millions', 'N/A'):,.1f} Million USD.", "KPP", ""])
    writer.writerow(["KPP-002", "Max Revisit Time", f"The mean revisit time shall be no more than {constraints.get('max_revisit', 'N/A'):.2f} hours.", "KPP", ""])
    writer.writerow(["KPP-003", "Min Image Quality", f"The system shall achieve a minimum mean image quality score of {constraints.get('min_quality', 'N/A'):.4f}.", "KPP", ""])
    writer.writerow(["PERF-001", "Target Cost", f"The implemented system shall have a target cost of ${optimal_design['cost']/1_000_000:,.1f}M USD.", "Performance", "KPP-001"])
    writer.writerow(["PERF-002", "Target Revisit Time", f"The system shall be designed to achieve a mean revisit time of {optimal_design['revisit_time']:.2f} hours.", "Performance", "KPP-002"])
    writer.writerow(["PERF-003", "Target Image Quality", f"The system shall be designed to achieve a mean image quality of {optimal_design['quality']:.4f}.", "Performance", "KPP-003"])
    
    num_planes = get_int_param_from_series(optimal_design, 'num_planes')
    arch_id_planes = "ARCH-001"
    writer.writerow([arch_id_planes, "Number of Planes", f"The constellation shall consist of {num_planes} orbital planes.", "Architectural", "PERF-002;PERF-003"])
    
    for i in range(1, num_planes + 1):
        plane_id = f"ARCH-P{i}"
        writer.writerow([f"{plane_id}-01", f"Plane {i} Altitude", f"Plane {i} shall have a nominal altitude of {optimal_design.get(f'plane_{i}_alt'):.1f} km.", "Architectural", arch_id_planes])
        writer.writerow([f"{plane_id}-02", f"Plane {i} Inclination", f"Plane {i} shall have a nominal inclination of {optimal_design.get(f'plane_{i}_inc'):.1f} degrees.", "Architectural", arch_id_planes])
        hq = get_int_param_from_series(optimal_design, f"plane_{i}_high_q")
        mq = get_int_param_from_series(optimal_design, f"plane_{i}_med_q")
        lq = get_int_param_from_series(optimal_design, f"plane_{i}_low_q")
        comp_id = f"{plane_id}-03"
        writer.writerow([comp_id, f"Plane {i} Satellite Complement", f"Plane {i} shall contain {hq} high-q, {mq} med-q, and {lq} low-q satellites.", "Architectural", arch_id_planes])

    return output.getvalue()


def _generate_uuid():
    """Generates a unique ID string."""
    return str(uuid.uuid4())

def generate_bdd_dot(optimal_design: pd.Series, system_name: str = "OptimalSystemDesign") -> str:
    """ # OptimalSystemDesign
    Generates a DOT language string for a Block Definition Diagram (BDD)
    representing the system architecture.
    """
    if optimal_design is None or optimal_design.empty:
        # st.warning("No optimal design selected. Cannot generate BDD DOT string.") # Cannot use st here
        return ""

    dot_lines = [
        "digraph BDD {",
        "  rankdir=TB; // Top to Bottom layout",
        "  node [shape=record, style=filled, fillcolor=\"#FFF9C4\", fontname=\"Arial\", fontsize=10, penwidth=1.5];", # Light yellow fill
        "  edge [fontname=\"Arial\", fontsize=9, penwidth=1.5];",
        "  graph [fontname=\"Arial\", fontsize=12, label=\"System Architecture - Block Definition Diagram\"];",
        "",
        "  // Block Definitions",
    ]

    # Satellite Type Blocks
    satellite_types_attrs = {
        "HighQualitySatellite": ["cost: Real", "massKg: Real", "sensorQuality: Real"],
        "MediumQualitySatellite": ["cost: Real", "massKg: Real", "sensorQuality: Real"],
        "LowQualitySatellite": ["cost: Real", "massKg: Real", "sensorQuality: Real"],
    }
    for sat_type, attrs in satellite_types_attrs.items():
        attrs_str = "\\l".join(attrs) + "\\l"
        dot_lines.append(f'  {sat_type} [label="{{<b>{sat_type}</b>|{attrs_str}}}"];') # Bold block name
    dot_lines.append("")

    # OrbitalPlane Block
    op_attrs = [
        "altitudeKm: Real", "inclinationDeg: Real",
        "numHighQ: Integer", "numMedQ: Integer", "numLowQ: Integer"
    ]
    op_attrs_str = "\\l".join(op_attrs) + "\\l"
    dot_lines.append(f'  OrbitalPlane [label="{{<b>OrbitalPlane</b>|{op_attrs_str}}}"];')
    dot_lines.append("")

    # GroundSegment Block
    gs_attrs = [
        "location: String", "uplinkCapacityMbps: Real", "downlinkCapacityMbps: Real"
    ]
    gs_attrs_str = "\\l".join(gs_attrs) + "\\l"
    dot_lines.append(f'  GroundSegment [label="{{<b>GroundSegment</b>|{gs_attrs_str}}}"];')
    dot_lines.append("")

    # System Block (representing the specific design context, e.g., OptimalDesign)
    num_planes = get_int_param_from_series(optimal_design, 'num_planes')
    sys_attrs_values = [
        f"totalCostUSD: {optimal_design.get('cost', 0.0):.2f}",
        f"meanRevisitTimeHr: {optimal_design.get('revisit_time', 0.0):.2f}",
        f"meanAchievedQuality: {optimal_design.get('quality', 0.0):.4f}",
        f"numPlanes: {num_planes}"
    ]
    if 'f_phasing' in optimal_design and pd.notna(optimal_design['f_phasing']):
        sys_attrs_values.append(f"fPhasing: {get_int_param_from_series(optimal_design, 'f_phasing')}")

    sys_attrs_str = "\\l".join(sys_attrs_values) + "\\l"
    dot_lines.append(f'  {system_name} [label="{{<b>{system_name}</b> (System Context)|{sys_attrs_str}}}"];') # Clarify it's the specific system context
    dot_lines.append("")

    dot_lines.append("  // Relationships")
    dot_lines.append(f'  {system_name} -> OrbitalPlane [arrowhead=odiamond, taillabel="1", headlabel="1..{num_planes}", label=" consists of"];')
    dot_lines.append(f'  OrbitalPlane -> HighQualitySatellite [arrowhead=odiamond, taillabel="1", headlabel="0..*", label=" contains (numHighQ)"];')
    dot_lines.append(f'  OrbitalPlane -> MediumQualitySatellite [arrowhead=odiamond, taillabel="1", headlabel="0..*", label=" contains (numMedQ)"];')
    dot_lines.append(f'  OrbitalPlane -> LowQualitySatellite [arrowhead=odiamond, taillabel="1", headlabel="0..*", label=" contains (numLowQ)"];')
    dot_lines.append(f'  {system_name} -> GroundSegment [arrowhead=vee, label=" interfaces with"];') # Association
    # The following were added based on user request in a previous turn, ensure they are present
    dot_lines.append(f'  UserSegment [label="{{<b>UserSegment</b>|userType: String\\ldataRequirements: String\\lserviceLevelAgreement: String\\l}}"];')
    dot_lines.append(f'  {system_name} -> UserSegment [arrowhead=vee, label=" provides service to"];')
    dot_lines.append(f'  GroundSegment -> UserSegment [arrowhead=vee, label=" disseminates data to"];')
    dot_lines.append("}")
    return "\n".join(dot_lines)

def generate_xmi_model(optimal_design: pd.Series, filename: str = "system_model") -> str:
    """
    Generates a SysML v2-compliant XMI file using standard XML libraries, without pyecore.
    """
    if optimal_design is None or optimal_design.empty:
        st.warning("No optimal design selected. Cannot generate XMI model.")
        return None
    
    if SATELLITE_OPTIONS is None:
        st.error("SATELLITE_OPTIONS from constellation_sim.py could not be loaded. Cannot generate full XMI model.")
        return None

    # --- XML Namespace and Root Element Setup ---
    # Define namespaces for XMI, UML, and SysML
    ns = {
        'xmi': 'http://www.omg.org/spec/XMI/20131001',
        'uml': 'http://www.eclipse.org/uml2/5.0.0/UML',
        'sysml': 'http://www.eclipse.org/papyrus/0.7.0/SysML/Blocks' # Namespace for the Block stereotype
    }
    for prefix, uri in ns.items():
        ET.register_namespace(prefix, uri)

    # Create root XMI element
    root = ET.Element(f'{{{ns["xmi"]}}}XMI')
    
    # --- Create Model Structure ---
    model_root = ET.SubElement(root, f'{{{ns["uml"]}}}Model', {'xmi:id': _generate_uuid(), 'name': 'SystemModel'})
    logical_pkg = ET.SubElement(model_root, 'packagedElement', {'xmi:type': 'uml:Package', 'xmi:id': _generate_uuid(), 'name': 'Logical'})

    # --- Helper to create a SysML Block ---
    # In XMI, a SysML Block is a UML Class with a 'Block' stereotype applied to it.
    def create_sysml_block(name, parent_package_element):
        block_id = _generate_uuid()
        # 1. Create the UML Class element that will be stereotyped
        uml_class = ET.SubElement(parent_package_element, 'packagedElement', {
            'xmi:type': 'uml:Class',
            'xmi:id': block_id,
            'name': name
        })
        
        # 2. Create the stereotype element itself
        # This assumes the stereotype application is a root-level element pointing to the base class.
        # Different tools might expect this differently. This is a common pattern.
        stereotype_app = ET.SubElement(root, f'{{{ns["sysml"]}}}Block', {
            'base_Class': block_id
        })
        return uml_class

    # --- Helper to create a Property on a Block ---
    def create_property(block_element, name, type_name, value=None, aggregation='none'):
        prop_attrs = {
            'xmi:type': 'uml:Property',
            'xmi:id': _generate_uuid(),
            'name': name,
            'aggregation': aggregation
        }
        # SysML and UML use standard XSD types. We can reference them by name.
        # This is simpler than creating PrimitiveType elements unless you need custom types.
        type_id_map = {
            'Real': 'pathmap://UML_LIBRARIES/EcorePrimitiveTypes.library.uml#EFloat',
            'Integer': 'pathmap://UML_LIBRARIES/EcorePrimitiveTypes.library.uml#EInt',
            'String': 'pathmap://UML_LIBRARIES/EcorePrimitiveTypes.library.uml#EString'
        }
        if type_name in type_id_map:
            prop_attrs['type'] = type_id_map[type_name]

        prop = ET.SubElement(block_element, 'ownedAttribute', prop_attrs)

        if value is not None:
            # Create a defaultValue element based on the type
            value_type_map = {
                'Real': 'uml:LiteralReal',
                'Integer': 'uml:LiteralInteger',
                'String': 'uml:LiteralString'
            }
            if type_name in value_type_map:
                ET.SubElement(prop, 'defaultValue', {
                    'xmi:type': value_type_map[type_name],
                    'xmi:id': _generate_uuid(),
                    'value': str(value)
                })
        return prop

    # Helper function to create associations and their end properties on blocks
    def create_association_and_ends(package_element, assoc_name,
                                    block_a_element, role_name_on_a,
                                    block_b_element, role_name_on_b,
                                    multiplicity_at_b_end="0..1", multiplicity_at_a_end="0..1"): # Multiplicity of the role on A (points to B), and role on B (points to A)
        assoc_id = _generate_uuid()
        assoc = ET.SubElement(package_element, 'packagedElement', {
            'xmi:type': 'uml:Association',
            'xmi:id': assoc_id,
            'name': assoc_name
        })

        # Property on Block A, typed by Block B (represents end connected to B)
        prop_on_a_id = _generate_uuid()
        prop_on_a = ET.SubElement(block_a_element, 'ownedAttribute', {
            'xmi:type': 'uml:Property', 'xmi:id': prop_on_a_id,
            'name': role_name_on_a, 'type': block_b_element.get('xmi:id'),
            'association': assoc_id
        })
        # Multiplicity for the end at Block B (defined by property on Block A)
        _add_multiplicity_to_property(prop_on_a, multiplicity_at_b_end)

        # Property on Block B, typed by Block A (represents end connected to A)
        prop_on_b_id = _generate_uuid()
        prop_on_b = ET.SubElement(block_b_element, 'ownedAttribute', {
            'xmi:type': 'uml:Property', 'xmi:id': prop_on_b_id,
            'name': role_name_on_b, 'type': block_a_element.get('xmi:id'),
            'association': assoc_id
        })
        # Multiplicity for the end at Block A (defined by property on Block B)
        _add_multiplicity_to_property(prop_on_b, multiplicity_at_a_end)

        assoc.set('memberEnd', f"{prop_on_a_id} {prop_on_b_id}")
        return assoc

    def _add_multiplicity_to_property(prop_element, multiplicity_str):
        parts = str(multiplicity_str).split('..')
        lower_val = parts[0]
        upper_val = parts[1] if len(parts) > 1 else lower_val

        ET.SubElement(prop_element, 'lowerValue', {'xmi:type': 'uml:LiteralInteger', 'xmi:id': _generate_uuid(), 'value': lower_val})
        if upper_val == '*':
            ET.SubElement(prop_element, 'upperValue', {'xmi:type': 'uml:LiteralUnlimitedNatural', 'xmi:id': _generate_uuid(), 'value': '*'})
        else:
            ET.SubElement(prop_element, 'upperValue', {'xmi:type': 'uml:LiteralInteger', 'xmi:id': _generate_uuid(), 'value': upper_val})

    # --- Define Block Definitions (Types) ---
    constellation_type_def = create_sysml_block('ConstellationType', logical_pkg)
    plane_type_def = create_sysml_block('OrbitalPlaneType', logical_pkg)

    # Define attributes of the OrbitalPlaneType
    create_property(plane_type_def, 'altitudeKm', 'Real')
    create_property(plane_type_def, 'inclinationDeg', 'Real')
    create_property(plane_type_def, 'numSatellitesInPlane', 'Integer')
    create_property(plane_type_def, 'numHighQualitySatellites', 'Integer')
    create_property(plane_type_def, 'numMediumQualitySatellites', 'Integer')
    create_property(plane_type_def, 'numLowQualitySatellites', 'Integer')

    # Define GroundSegmentType Block
    ground_segment_type_def = create_sysml_block('GroundSegmentType', logical_pkg)
    create_property(ground_segment_type_def, 'location', 'String', value="DefaultLocation")
    create_property(ground_segment_type_def, 'uplinkCapacityMbps', 'Real', value="100.0")
    create_property(ground_segment_type_def, 'downlinkCapacityMbps', 'Real', value="1000.0")

    # Define UserSegmentType Block
    user_segment_type_def = create_sysml_block('UserSegmentType', logical_pkg)
    create_property(user_segment_type_def, 'userType', 'String', value="GenericUser")
    create_property(user_segment_type_def, 'dataRequirements', 'String', value="StandardDataProducts")

    # --- Define Specific Satellite Type Blocks ---
    satellite_keys_map = {
        'LowQualitySatellite': 'low_quality',
        'MediumQualitySatellite': 'medium_quality',
        'HighQualitySatellite': 'high_quality'
    }

    for block_name, sim_key in satellite_keys_map.items():
        sat_class = create_sysml_block(block_name, logical_pkg)
        options = SATELLITE_OPTIONS.get(sim_key, {})
        create_property(sat_class, 'cost', 'Real', value=float(options.get('cost', 0.0)))
        create_property(sat_class, 'massKg', 'Real', value=float(options.get('mass', 0.0)))
        create_property(sat_class, 'sensorQuality', 'Real', value=float(options.get('sensor_quality', 0.0)))

    # --- Create the Main System Block representing the specific design ---
    actual_constellation_block = create_sysml_block(filename, logical_pkg)
    
    # --- Populate Properties of the Actual Constellation ---
    num_planes = get_int_param_from_series(optimal_design, 'num_planes')
    total_sats = sum(
        get_int_param_from_series(optimal_design, f"plane_{i}_high_q", 0) +
        get_int_param_from_series(optimal_design, f"plane_{i}_med_q", 0) +
        get_int_param_from_series(optimal_design, f"plane_{i}_low_q", 0)
        for i in range(1, num_planes + 1)
    )
    
    create_property(actual_constellation_block, 'numPlanes', 'Integer', value=num_planes)
    create_property(actual_constellation_block, 'numSatellites', 'Integer', value=total_sats)
    create_property(actual_constellation_block, 'totalCostUSD', 'Real', value=f"{optimal_design.get('cost', 0.0):.2f}")
    create_property(actual_constellation_block, 'meanRevisitTimeHr', 'Real', value=f"{optimal_design.get('revisit_time', 0.0):.2f}")
    create_property(actual_constellation_block, 'meanAchievedQuality', 'Real', value=f"{optimal_design.get('quality', 0.0):.4f}")

    if 'f_phasing' in optimal_design and pd.notna(optimal_design['f_phasing']):
        create_property(actual_constellation_block, 'fPhasing', 'Integer', value=get_int_param_from_series(optimal_design, 'f_phasing'))

    # --- Create Part Properties for Each Orbital Plane ---
    for i in range(1, num_planes + 1):
        # Create a part property representing the plane, typed by OrbitalPlaneType
        plane_part = create_property(actual_constellation_block, f'plane{i}', 'OrbitalPlaneType', aggregation='composite')
        
        # Here we would create an InstanceSpecification with slots to hold the values,
        # but for simplicity and compatibility with many tools, defining a specific Block for each
        # plane instance is often more straightforward.
        
        # Create a specific Block definition for this instance of a plane
        plane_instance_block = create_sysml_block(f'Plane_{i}', logical_pkg)
        
        alt_p = optimal_design.get(f'plane_{i}_alt', 0.0)
        inc_p = optimal_design.get(f'plane_{i}_inc', 0.0)
        hq_p = get_int_param_from_series(optimal_design, f"plane_{i}_high_q", 0)
        mq_p = get_int_param_from_series(optimal_design, f"plane_{i}_med_q", 0)
        lq_p = get_int_param_from_series(optimal_design, f"plane_{i}_low_q", 0)
        sats_in_plane_p = hq_p + mq_p + lq_p

        # Add properties with values to this specific plane's Block definition
        create_property(plane_instance_block, 'altitudeKm', 'Real', value=f"{alt_p:.1f}")
        create_property(plane_instance_block, 'inclinationDeg', 'Real', value=f"{inc_p:.1f}")
        create_property(plane_instance_block, 'numSatellitesInPlane', 'Integer', value=sats_in_plane_p)
        create_property(plane_instance_block, 'numHighQualitySatellites', 'Integer', value=hq_p)
        create_property(plane_instance_block, 'numMediumQualitySatellites', 'Integer', value=mq_p)
        create_property(plane_instance_block, 'numLowQualitySatellites', 'Integer', value=lq_p)

    # --- Define Associations based on BDD ---
    # Association: System <-> GroundSegment
    create_association_and_ends(logical_pkg, "InterfacesWith",
                                actual_constellation_block, "connectedGroundSegment",
                                ground_segment_type_def, "interfacingSystem",
                                multiplicity_at_b_end="0..1", multiplicity_at_a_end="0..*") # System can interface with 0..1 GS, GS can interface with 0..* Systems

    # Association: System <-> UserSegment
    create_association_and_ends(logical_pkg, "ProvidesServiceTo",
                                actual_constellation_block, "servicedUserSegment",
                                user_segment_type_def, "serviceProvidingSystem",
                                multiplicity_at_b_end="0..*", multiplicity_at_a_end="0..1") # System can service 0..* US, US serviced by 0..1 System

    # Association: GroundSegment <-> UserSegment
    create_association_and_ends(logical_pkg, "DisseminatesDataTo",
                                ground_segment_type_def, "dataRecipientUserSegment",
                                user_segment_type_def, "dataDisseminatingGroundSegment",
                                multiplicity_at_b_end="0..*", multiplicity_at_a_end="0..*") # GS can disseminate to 0..* US, US can receive from 0..* GS


    # --- Save the Model to an XMI file ---
    try:
        # Pretty print the XML
        xml_string = ET.tostring(root, 'utf-8')
        parsed_string = minidom.parseString(xml_string)
        pretty_xml_string = parsed_string.toprettyxml(indent="  ")

        output_filename = f"{filename}.xmi"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(pretty_xml_string)
            
        st.success(f"Successfully generated XMI model at '{output_filename}'")
        return output_filename
        
    except Exception as e:
        st.error(f"An error occurred while generating the XMI model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None
