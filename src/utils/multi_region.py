"""
Multi-region deployment utilities for global liquid neural framework deployment.

Supports deployment across multiple cloud regions with automatic failover,
load balancing, and data residency compliance.
"""

import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings


class RegionStatus(Enum):
    """Status of deployment regions."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region_id: str
    region_name: str
    cloud_provider: str
    endpoint_url: str
    data_residency_rules: List[str]
    compliance_frameworks: List[str]
    supported_languages: List[str]
    compute_capacity: Dict[str, Any]
    status: RegionStatus = RegionStatus.ACTIVE
    latency_ms: float = 0.0
    last_health_check: float = 0.0


class MultiRegionManager:
    """Manager for multi-region deployment and operations."""
    
    def __init__(self):
        self.regions = {}
        self.region_weights = {}
        self.failover_chains = {}
        self.load_balancer_config = {}
        
        # Initialize default regions
        self._initialize_default_regions()
    
    def _initialize_default_regions(self):
        """Initialize default global regions."""
        default_regions = [
            RegionConfig(
                region_id="us-east-1",
                region_name="US East (N. Virginia)",
                cloud_provider="AWS",
                endpoint_url="https://liquid-neural-us-east-1.example.com",
                data_residency_rules=["US", "North America"],
                compliance_frameworks=["SOC2", "CCPA", "HIPAA"],
                supported_languages=["en", "es", "fr"],
                compute_capacity={"max_models": 100, "max_batch_size": 1000},
                latency_ms=50.0
            ),
            RegionConfig(
                region_id="eu-west-1", 
                region_name="Europe (Ireland)",
                cloud_provider="AWS",
                endpoint_url="https://liquid-neural-eu-west-1.example.com",
                data_residency_rules=["EU", "EEA"],
                compliance_frameworks=["GDPR", "ISO27001"],
                supported_languages=["en", "de", "fr", "es"],
                compute_capacity={"max_models": 80, "max_batch_size": 800},
                latency_ms=45.0
            ),
            RegionConfig(
                region_id="ap-southeast-1",
                region_name="Asia Pacific (Singapore)", 
                cloud_provider="AWS",
                endpoint_url="https://liquid-neural-ap-southeast-1.example.com",
                data_residency_rules=["Singapore", "ASEAN"],
                compliance_frameworks=["PDPA", "ISO27001"],
                supported_languages=["en", "zh", "ja"],
                compute_capacity={"max_models": 60, "max_batch_size": 600},
                latency_ms=60.0
            ),
            RegionConfig(
                region_id="eu-central-1",
                region_name="Europe (Frankfurt)",
                cloud_provider="AWS", 
                endpoint_url="https://liquid-neural-eu-central-1.example.com",
                data_residency_rules=["Germany", "EU"],
                compliance_frameworks=["GDPR", "BSI C5"],
                supported_languages=["de", "en", "fr"],
                compute_capacity={"max_models": 70, "max_batch_size": 700},
                latency_ms=40.0
            ),
            RegionConfig(
                region_id="ap-northeast-1",
                region_name="Asia Pacific (Tokyo)",
                cloud_provider="AWS",
                endpoint_url="https://liquid-neural-ap-northeast-1.example.com", 
                data_residency_rules=["Japan"],
                compliance_frameworks=["APPI", "ISO27001"],
                supported_languages=["ja", "en"],
                compute_capacity={"max_models": 50, "max_batch_size": 500},
                latency_ms=55.0
            )
        ]
        
        # Register regions
        for region in default_regions:
            self.register_region(region)
        
        # Set up failover chains
        self.failover_chains = {
            "us-east-1": ["us-west-2", "eu-west-1"],
            "eu-west-1": ["eu-central-1", "us-east-1"], 
            "ap-southeast-1": ["ap-northeast-1", "us-east-1"],
            "eu-central-1": ["eu-west-1", "us-east-1"],
            "ap-northeast-1": ["ap-southeast-1", "us-east-1"]
        }
        
        # Initialize load balancer weights (based on capacity and latency)
        self._calculate_load_balancer_weights()
    
    def register_region(self, region: RegionConfig):
        """Register a new deployment region."""
        self.regions[region.region_id] = region
        
        # Set initial weight based on compute capacity and latency
        capacity_score = region.compute_capacity.get("max_models", 50)
        latency_score = max(100 - region.latency_ms, 10)  # Lower latency = higher score
        
        self.region_weights[region.region_id] = (capacity_score * latency_score) / 1000
        
        print(f"✅ Registered region {region.region_name} ({region.region_id})")
    
    def _calculate_load_balancer_weights(self):
        """Calculate load balancer weights based on region capabilities."""
        total_weight = 0
        
        for region_id, region in self.regions.items():
            if region.status == RegionStatus.ACTIVE:
                # Weight based on capacity, latency, and health
                capacity_factor = region.compute_capacity.get("max_models", 50) / 100
                latency_factor = max(200 - region.latency_ms, 50) / 200
                health_factor = 1.0 if region.status == RegionStatus.ACTIVE else 0.1
                
                weight = capacity_factor * latency_factor * health_factor
                self.region_weights[region_id] = weight
                total_weight += weight
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            for region_id in self.region_weights:
                self.region_weights[region_id] /= total_weight
        
        # Update load balancer configuration
        self.load_balancer_config = {
            'algorithm': 'weighted_round_robin',
            'weights': self.region_weights.copy(),
            'health_check_interval': 30,
            'failover_threshold': 0.8
        }
    
    def select_optimal_region(self, 
                            user_location: Optional[str] = None,
                            data_residency_requirement: Optional[str] = None,
                            compliance_requirements: Optional[List[str]] = None) -> str:
        """Select optimal region based on user requirements."""
        
        candidate_regions = []
        
        for region_id, region in self.regions.items():
            if region.status != RegionStatus.ACTIVE:
                continue
            
            # Check data residency requirements
            if data_residency_requirement:
                if data_residency_requirement not in region.data_residency_rules:
                    continue
            
            # Check compliance requirements
            if compliance_requirements:
                if not any(req in region.compliance_frameworks for req in compliance_requirements):
                    continue
            
            # Calculate score based on multiple factors
            score = 0.0
            
            # Latency score (lower is better)
            latency_score = max(200 - region.latency_ms, 10) / 200
            score += latency_score * 0.4
            
            # Capacity score
            capacity_score = region.compute_capacity.get("max_models", 50) / 100
            score += capacity_score * 0.3
            
            # Geographic proximity (simplified)
            if user_location:
                proximity_score = self._calculate_proximity_score(user_location, region.region_id)
                score += proximity_score * 0.3
            
            candidate_regions.append((region_id, score))
        
        if not candidate_regions:
            # Fall back to any active region
            active_regions = [r for r, reg in self.regions.items() if reg.status == RegionStatus.ACTIVE]
            return active_regions[0] if active_regions else list(self.regions.keys())[0]
        
        # Select region with highest score
        candidate_regions.sort(key=lambda x: x[1], reverse=True)
        return candidate_regions[0][0]
    
    def _calculate_proximity_score(self, user_location: str, region_id: str) -> float:
        """Calculate geographic proximity score (simplified)."""
        
        # Simplified geographic proximity mapping
        proximity_map = {
            'US': {'us-east-1': 1.0, 'us-west-2': 0.9, 'eu-west-1': 0.3, 'ap-southeast-1': 0.2},
            'Canada': {'us-east-1': 0.9, 'us-west-2': 0.8, 'eu-west-1': 0.4, 'ap-southeast-1': 0.2},
            'UK': {'eu-west-1': 1.0, 'eu-central-1': 0.9, 'us-east-1': 0.4, 'ap-southeast-1': 0.2},
            'Germany': {'eu-central-1': 1.0, 'eu-west-1': 0.9, 'us-east-1': 0.3, 'ap-southeast-1': 0.2},
            'Singapore': {'ap-southeast-1': 1.0, 'ap-northeast-1': 0.8, 'eu-west-1': 0.3, 'us-east-1': 0.3},
            'Japan': {'ap-northeast-1': 1.0, 'ap-southeast-1': 0.8, 'us-east-1': 0.4, 'eu-west-1': 0.2},
            'France': {'eu-west-1': 0.9, 'eu-central-1': 0.8, 'us-east-1': 0.3, 'ap-southeast-1': 0.2}
        }
        
        return proximity_map.get(user_location, {}).get(region_id, 0.5)
    
    def get_region_health(self, region_id: str) -> Dict[str, Any]:
        """Get health status of a specific region."""
        
        if region_id not in self.regions:
            return {'error': 'Region not found'}
        
        region = self.regions[region_id]
        
        # Simulate health metrics (in real implementation, would query actual services)
        health_metrics = {
            'region_id': region_id,
            'status': region.status.value,
            'latency_ms': region.latency_ms,
            'cpu_utilization': 65.0,  # Simulated
            'memory_utilization': 78.0,  # Simulated
            'active_models': 42,  # Simulated
            'requests_per_second': 150.0,  # Simulated
            'error_rate': 0.02,  # Simulated
            'last_health_check': time.time(),
            'uptime_percentage': 99.9
        }
        
        # Update region with simulated health check
        region.last_health_check = time.time()
        
        return health_metrics
    
    def perform_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all regions."""
        
        health_results = {}
        
        for region_id in self.regions:
            health_results[region_id] = self.get_region_health(region_id)
            
            # Update region status based on health
            health = health_results[region_id]
            
            if 'error' in health:
                self.regions[region_id].status = RegionStatus.OFFLINE
            elif health.get('error_rate', 0) > 0.05:  # >5% error rate
                self.regions[region_id].status = RegionStatus.DEGRADED
            else:
                self.regions[region_id].status = RegionStatus.ACTIVE
        
        # Recalculate load balancer weights based on updated health
        self._calculate_load_balancer_weights()
        
        return health_results
    
    def trigger_failover(self, failed_region: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger failover to backup region."""
        
        failover_result = {
            'failed_region': failed_region,
            'failover_triggered': time.time(),
            'backup_regions': [],
            'selected_backup': None,
            'failover_successful': False,
            'response_time_ms': 0.0
        }
        
        # Get failover chain for failed region
        backup_candidates = self.failover_chains.get(failed_region, [])
        
        # Filter to only active backup regions
        active_backups = []
        for backup_region in backup_candidates:
            if backup_region in self.regions and self.regions[backup_region].status == RegionStatus.ACTIVE:
                active_backups.append(backup_region)
        
        failover_result['backup_regions'] = active_backups
        
        if not active_backups:
            # No healthy backup regions available
            failover_result['error'] = 'No healthy backup regions available'
            return failover_result
        
        # Select best backup region based on compliance and capacity
        selected_backup = None
        
        # Check data residency and compliance requirements
        original_region = self.regions[failed_region]
        
        for backup_region in active_backups:
            backup_config = self.regions[backup_region]
            
            # Check if backup region can handle the same compliance requirements
            compliance_compatible = any(
                framework in backup_config.compliance_frameworks 
                for framework in original_region.compliance_frameworks
            )
            
            # Check data residency compatibility
            residency_compatible = any(
                rule in backup_config.data_residency_rules
                for rule in original_region.data_residency_rules
            )
            
            if compliance_compatible and residency_compatible:
                selected_backup = backup_region
                break
        
        # If no fully compatible backup, select first available
        if not selected_backup:
            selected_backup = active_backups[0]
            warnings.warn(f"Failover to {selected_backup} may not meet all compliance requirements")
        
        failover_result['selected_backup'] = selected_backup
        
        # Simulate failover process
        start_time = time.time()
        
        try:
            # In real implementation, would:
            # 1. Route traffic to backup region
            # 2. Sync any necessary state
            # 3. Update load balancer configuration
            # 4. Notify monitoring systems
            
            # Simulate processing time
            import random
            processing_time = random.uniform(100, 500)  # 100-500ms
            
            failover_result.update({
                'failover_successful': True,
                'response_time_ms': processing_time,
                'backup_endpoint': self.regions[selected_backup].endpoint_url
            })
            
        except Exception as e:
            failover_result.update({
                'failover_successful': False,
                'error': str(e),
                'response_time_ms': (time.time() - start_time) * 1000
            })
        
        return failover_result
    
    def get_deployment_topology(self) -> Dict[str, Any]:
        """Get current multi-region deployment topology."""
        
        topology = {
            'total_regions': len(self.regions),
            'active_regions': sum(1 for r in self.regions.values() if r.status == RegionStatus.ACTIVE),
            'regions': {},
            'load_balancer': self.load_balancer_config,
            'failover_chains': self.failover_chains,
            'compliance_coverage': {},
            'language_support': {},
            'total_capacity': {
                'max_models': sum(r.compute_capacity.get('max_models', 0) for r in self.regions.values()),
                'max_batch_size': sum(r.compute_capacity.get('max_batch_size', 0) for r in self.regions.values())
            }
        }
        
        # Detailed region information
        for region_id, region in self.regions.items():
            topology['regions'][region_id] = {
                'name': region.region_name,
                'status': region.status.value,
                'cloud_provider': region.cloud_provider,
                'latency_ms': region.latency_ms,
                'weight': self.region_weights.get(region_id, 0.0),
                'compliance_frameworks': region.compliance_frameworks,
                'supported_languages': region.supported_languages,
                'data_residency_rules': region.data_residency_rules,
                'compute_capacity': region.compute_capacity
            }
        
        # Aggregate compliance coverage
        all_frameworks = set()
        for region in self.regions.values():
            all_frameworks.update(region.compliance_frameworks)
        
        for framework in all_frameworks:
            supporting_regions = [
                r.region_id for r in self.regions.values() 
                if framework in r.compliance_frameworks and r.status == RegionStatus.ACTIVE
            ]
            topology['compliance_coverage'][framework] = supporting_regions
        
        # Aggregate language support
        all_languages = set()
        for region in self.regions.values():
            all_languages.update(region.supported_languages)
        
        for language in all_languages:
            supporting_regions = [
                r.region_id for r in self.regions.values()
                if language in r.supported_languages and r.status == RegionStatus.ACTIVE
            ]
            topology['language_support'][language] = supporting_regions
        
        return topology
    
    def optimize_deployment(self) -> Dict[str, Any]:
        """Optimize multi-region deployment based on current usage patterns."""
        
        optimization_results = {
            'timestamp': time.time(),
            'current_topology': self.get_deployment_topology(),
            'recommendations': [],
            'cost_optimization': {},
            'performance_optimization': {},
            'compliance_optimization': {}
        }
        
        # Analyze current deployment
        active_regions = [r for r in self.regions.values() if r.status == RegionStatus.ACTIVE]
        
        # Cost optimization recommendations
        total_capacity = sum(r.compute_capacity.get('max_models', 0) for r in active_regions)
        if total_capacity > 200:  # Arbitrary threshold
            optimization_results['cost_optimization']['recommendation'] = 'Consider scaling down underutilized regions'
            optimization_results['cost_optimization']['potential_savings'] = '15-25%'
        
        # Performance optimization recommendations
        avg_latency = sum(r.latency_ms for r in active_regions) / len(active_regions)
        if avg_latency > 60:
            optimization_results['performance_optimization']['recommendation'] = 'Consider adding regions closer to users'
            optimization_results['performance_optimization']['expected_improvement'] = f'Reduce latency by {avg_latency - 50}ms'
        
        # Compliance optimization
        compliance_gaps = []
        for framework in ['GDPR', 'CCPA', 'PDPA']:
            supporting_regions = [
                r for r in active_regions 
                if framework in r.compliance_frameworks
            ]
            if len(supporting_regions) < 2:  # Should have redundancy
                compliance_gaps.append(framework)
        
        if compliance_gaps:
            optimization_results['compliance_optimization']['gaps'] = compliance_gaps
            optimization_results['compliance_optimization']['recommendation'] = 'Add redundant regions for compliance frameworks'
        
        return optimization_results


# Global multi-region manager instance
multi_region_manager = MultiRegionManager()


# Convenience functions
def select_region(user_location: str = None, 
                 data_residency: str = None,
                 compliance_reqs: List[str] = None) -> str:
    """Select optimal region for request."""
    return multi_region_manager.select_optimal_region(
        user_location, data_residency, compliance_reqs
    )


def check_all_regions_health() -> Dict[str, Dict[str, Any]]:
    """Check health of all deployment regions."""
    return multi_region_manager.perform_health_checks()


def get_deployment_info() -> Dict[str, Any]:
    """Get current deployment topology information."""
    return multi_region_manager.get_deployment_topology()