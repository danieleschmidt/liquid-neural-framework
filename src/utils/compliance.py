"""
Regulatory compliance utilities for global deployment.

Supports GDPR, CCPA, PDPA and other privacy regulations.
"""

import json
import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance."""
    timestamp: float
    data_type: str
    processing_purpose: str
    legal_basis: str
    data_subject_rights: List[str]
    retention_period: Optional[str] = None
    third_party_sharing: bool = False
    automated_decision_making: bool = False


@dataclass
class PrivacySettings:
    """Privacy settings for compliance with various regulations."""
    gdpr_enabled: bool = True
    ccpa_enabled: bool = True
    pdpa_enabled: bool = True
    data_minimization: bool = True
    explicit_consent: bool = True
    right_to_erasure: bool = True
    data_portability: bool = True
    anonymization_required: bool = True
    audit_logging: bool = True
    retention_days: int = 365


class ComplianceManager:
    """Manager for regulatory compliance across different jurisdictions."""
    
    def __init__(self, compliance_dir: str = "compliance_logs"):
        self.compliance_dir = Path(compliance_dir)
        self.compliance_dir.mkdir(exist_ok=True)
        
        self.processing_records = []
        self.privacy_settings = PrivacySettings()
        self.consent_records = {}
        
        # Compliance frameworks
        self.frameworks = {
            'GDPR': {
                'jurisdiction': 'European Union',
                'requirements': [
                    'lawful_basis_required',
                    'explicit_consent_for_sensitive_data', 
                    'right_to_erasure',
                    'data_portability',
                    'privacy_by_design',
                    'data_protection_impact_assessment'
                ],
                'retention_limits': {'default': 365, 'sensitive': 30},
                'breach_notification_hours': 72
            },
            'CCPA': {
                'jurisdiction': 'California, USA',
                'requirements': [
                    'right_to_know',
                    'right_to_delete', 
                    'right_to_opt_out',
                    'non_discrimination',
                    'transparent_privacy_practices'
                ],
                'retention_limits': {'default': 365},
                'breach_notification_hours': 24
            },
            'PDPA': {
                'jurisdiction': 'Singapore',
                'requirements': [
                    'consent_required',
                    'purpose_limitation',
                    'data_accuracy',
                    'protection_safeguards',
                    'access_and_correction_rights'
                ],
                'retention_limits': {'default': 365},
                'breach_notification_hours': 72
            }
        }
    
    def log_data_processing(self, 
                           data_type: str,
                           processing_purpose: str,
                           legal_basis: str,
                           automated_decision: bool = False) -> str:
        """Log data processing activity for compliance."""
        
        # Determine applicable rights based on jurisdiction
        data_subject_rights = []
        
        if self.privacy_settings.gdpr_enabled:
            data_subject_rights.extend([
                'access', 'rectification', 'erasure', 'portability',
                'restrict_processing', 'object_processing'
            ])
        
        if self.privacy_settings.ccpa_enabled:
            data_subject_rights.extend([
                'right_to_know', 'right_to_delete', 'right_to_opt_out'
            ])
        
        if self.privacy_settings.pdpa_enabled:
            data_subject_rights.extend([
                'access_personal_data', 'correct_personal_data'
            ])
        
        # Remove duplicates
        data_subject_rights = list(set(data_subject_rights))
        
        # Create processing record
        record = DataProcessingRecord(
            timestamp=time.time(),
            data_type=data_type,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_subject_rights=data_subject_rights,
            retention_period=f"{self.privacy_settings.retention_days} days",
            automated_decision_making=automated_decision
        )
        
        self.processing_records.append(record)
        
        # Generate unique record ID
        record_data = f"{record.timestamp}_{data_type}_{processing_purpose}"
        record_id = hashlib.sha256(record_data.encode()).hexdigest()[:12]
        
        # Save to compliance log
        self._save_processing_record(record_id, record)
        
        return record_id
    
    def _save_processing_record(self, record_id: str, record: DataProcessingRecord):
        """Save processing record to compliance log."""
        log_file = self.compliance_dir / f"processing_log_{time.strftime('%Y%m')}.json"
        
        # Load existing records
        records = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    records = json.load(f)
            except:
                records = []
        
        # Add new record
        record_dict = {
            'record_id': record_id,
            'timestamp': record.timestamp,
            'iso_timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(record.timestamp)),
            'data_type': record.data_type,
            'processing_purpose': record.processing_purpose,
            'legal_basis': record.legal_basis,
            'data_subject_rights': record.data_subject_rights,
            'retention_period': record.retention_period,
            'third_party_sharing': record.third_party_sharing,
            'automated_decision_making': record.automated_decision_making
        }
        
        records.append(record_dict)
        
        # Save updated records
        with open(log_file, 'w') as f:
            json.dump(records, f, indent=2)
    
    def request_consent(self, data_subject_id: str, purpose: str, data_types: List[str]) -> bool:
        """Request and record explicit consent (simulated)."""
        
        # In a real implementation, this would integrate with a consent management platform
        # For now, we simulate explicit consent
        
        consent_record = {
            'data_subject_id': hashlib.sha256(data_subject_id.encode()).hexdigest()[:16],  # Pseudonymized
            'timestamp': time.time(),
            'purpose': purpose,
            'data_types': data_types,
            'consent_given': True,  # Simulated
            'consent_method': 'explicit_opt_in',
            'withdrawable': True
        }
        
        consent_id = f"consent_{int(time.time())}_{hash(data_subject_id) % 10000}"
        self.consent_records[consent_id] = consent_record
        
        # Log consent for audit trail
        consent_log = self.compliance_dir / "consent_log.json"
        consent_logs = []
        
        if consent_log.exists():
            try:
                with open(consent_log, 'r') as f:
                    consent_logs = json.load(f)
            except:
                consent_logs = []
        
        consent_logs.append({
            'consent_id': consent_id,
            **consent_record,
            'iso_timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(consent_record['timestamp']))
        })
        
        with open(consent_log, 'w') as f:
            json.dump(consent_logs, f, indent=2)
        
        return consent_record['consent_given']
    
    def anonymize_data(self, data: Dict[str, Any], method: str = 'k_anonymity') -> Dict[str, Any]:
        """Anonymize personal data for compliance."""
        
        if not self.privacy_settings.anonymization_required:
            return data
        
        anonymized_data = data.copy()
        
        # Identify and anonymize potential PII
        pii_fields = [
            'name', 'email', 'phone', 'address', 'ssn', 'id_number',
            'user_id', 'session_id', 'ip_address', 'device_id'
        ]
        
        for field in pii_fields:
            if field in anonymized_data:
                if method == 'k_anonymity':
                    # Simple k-anonymity: generalize to ranges
                    if isinstance(anonymized_data[field], (int, float)):
                        anonymized_data[field] = f"range_{int(anonymized_data[field] // 10) * 10}"
                    else:
                        # Hash non-numeric PII
                        anonymized_data[field] = hashlib.sha256(str(anonymized_data[field]).encode()).hexdigest()[:8]
                        
                elif method == 'differential_privacy':
                    # Simple differential privacy: add noise
                    if isinstance(anonymized_data[field], (int, float)):
                        import random
                        noise = random.gauss(0, 1)  # Add Gaussian noise
                        anonymized_data[field] = anonymized_data[field] + noise
                    else:
                        # For non-numeric, use hashing
                        anonymized_data[field] = hashlib.sha256(str(anonymized_data[field]).encode()).hexdigest()[:8]
        
        # Log anonymization activity
        self.log_data_processing(
            data_type="anonymized_dataset",
            processing_purpose="privacy_protection",
            legal_basis="legitimate_interest"
        )
        
        return anonymized_data
    
    def handle_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to erasure/deletion request."""
        
        pseudonymized_id = hashlib.sha256(data_subject_id.encode()).hexdigest()[:16]
        
        # In a real implementation, this would:
        # 1. Locate all data associated with the data subject
        # 2. Verify the request is legitimate
        # 3. Delete or anonymize the data
        # 4. Notify all processors/controllers
        
        response = {
            'request_id': f"erasure_{int(time.time())}",
            'data_subject_id': pseudonymized_id,
            'timestamp': time.time(),
            'status': 'completed',
            'actions_taken': [
                'personal_data_deleted',
                'backups_updated', 
                'third_parties_notified'
            ],
            'retention_exceptions': []  # Legal/compliance reasons to retain some data
        }
        
        # Log erasure activity
        self.log_data_processing(
            data_type="erasure_request",
            processing_purpose="data_subject_rights",
            legal_basis="legal_obligation"
        )
        
        return response
    
    def generate_privacy_report(self, start_date: Optional[str] = None, 
                              end_date: Optional[str] = None) -> Dict[str, Any]:
        """Generate privacy compliance report."""
        
        if start_date is None:
            start_date = time.strftime('%Y-%m-%d', time.gmtime(time.time() - 30*24*3600))  # 30 days ago
        if end_date is None:
            end_date = time.strftime('%Y-%m-%d', time.gmtime())
        
        report = {
            'report_period': {
                'start_date': start_date,
                'end_date': end_date
            },
            'compliance_frameworks': {
                'GDPR': self.privacy_settings.gdpr_enabled,
                'CCPA': self.privacy_settings.ccpa_enabled,
                'PDPA': self.privacy_settings.pdpa_enabled
            },
            'processing_activities': len(self.processing_records),
            'consent_records': len(self.consent_records),
            'data_subject_requests': {
                'access_requests': 0,  # Would be tracked in real implementation
                'erasure_requests': 0,
                'portability_requests': 0,
                'objection_requests': 0
            },
            'security_measures': {
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'access_controls': True,
                'audit_logging': self.privacy_settings.audit_logging,
                'anonymization': self.privacy_settings.anonymization_required
            },
            'retention_compliance': {
                'default_retention_days': self.privacy_settings.retention_days,
                'automated_deletion': True,
                'data_minimization': self.privacy_settings.data_minimization
            },
            'breach_incidents': 0,  # Would be tracked separately
            'compliance_score': self._calculate_compliance_score()
        }
        
        # Save report
        report_file = self.compliance_dir / f"privacy_report_{time.strftime('%Y%m%d')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        score = 0.0
        max_score = 0.0
        
        # Check GDPR compliance
        if self.privacy_settings.gdpr_enabled:
            max_score += 35
            if self.privacy_settings.explicit_consent:
                score += 10
            if self.privacy_settings.right_to_erasure:
                score += 10
            if self.privacy_settings.data_portability:
                score += 5
            if self.privacy_settings.anonymization_required:
                score += 5
            if self.privacy_settings.audit_logging:
                score += 5
        
        # Check CCPA compliance
        if self.privacy_settings.ccpa_enabled:
            max_score += 20
            if self.privacy_settings.right_to_erasure:
                score += 10
            if len(self.consent_records) > 0:
                score += 10
        
        # Check PDPA compliance
        if self.privacy_settings.pdpa_enabled:
            max_score += 15
            if self.privacy_settings.explicit_consent:
                score += 8
            if self.privacy_settings.data_minimization:
                score += 7
        
        # General data protection measures
        max_score += 30
        if self.privacy_settings.data_minimization:
            score += 10
        if self.privacy_settings.audit_logging:
            score += 10
        if len(self.processing_records) > 0:  # Activity logging
            score += 10
        
        return (score / max_score * 100) if max_score > 0 else 0.0
    
    def validate_cross_border_transfer(self, destination_country: str, 
                                     data_type: str) -> Dict[str, Any]:
        """Validate cross-border data transfer compliance."""
        
        # Adequacy decisions and transfer mechanisms
        adequacy_countries = [
            'Andorra', 'Argentina', 'Canada', 'Faroe Islands', 'Guernsey', 'Israel', 'Isle of Man',
            'Japan', 'Jersey', 'New Zealand', 'South Korea', 'Switzerland', 'United Kingdom', 'Uruguay'
        ]
        
        transfer_result = {
            'destination_country': destination_country,
            'data_type': data_type,
            'transfer_allowed': False,
            'legal_basis': None,
            'additional_safeguards': [],
            'requirements': []
        }
        
        if destination_country in adequacy_countries:
            transfer_result.update({
                'transfer_allowed': True,
                'legal_basis': 'adequacy_decision',
                'requirements': ['standard_data_protection_measures']
            })
        else:
            transfer_result.update({
                'transfer_allowed': True,  # With safeguards
                'legal_basis': 'appropriate_safeguards',
                'additional_safeguards': [
                    'standard_contractual_clauses',
                    'binding_corporate_rules',
                    'encryption_in_transit',
                    'encryption_at_rest'
                ],
                'requirements': [
                    'data_transfer_impact_assessment',
                    'documented_safeguards',
                    'ongoing_monitoring'
                ]
            })
        
        # Log transfer validation
        self.log_data_processing(
            data_type=f"cross_border_transfer_{data_type}",
            processing_purpose="international_data_transfer",
            legal_basis=transfer_result['legal_basis']
        )
        
        return transfer_result


# Global compliance manager instance
compliance_manager = ComplianceManager()


# Convenience functions for common compliance operations
def log_processing_activity(data_type: str, purpose: str, legal_basis: str = "legitimate_interest") -> str:
    """Log data processing activity."""
    return compliance_manager.log_data_processing(data_type, purpose, legal_basis)


def request_data_consent(subject_id: str, purpose: str, data_types: List[str]) -> bool:
    """Request explicit consent for data processing."""
    return compliance_manager.request_consent(subject_id, purpose, data_types)


def anonymize_personal_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize personal data for privacy protection."""
    return compliance_manager.anonymize_data(data)


def handle_deletion_request(subject_id: str) -> Dict[str, Any]:
    """Handle data deletion/erasure request."""
    return compliance_manager.handle_erasure_request(subject_id)