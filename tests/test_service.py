from app.services.waste_audit_service import WasteAuditService

def test_uncertain_input():
    service = WasteAuditService()
    result = service.generate_disposal_plan("")

    assert result["category"] == "Uncertain"
    assert "disposal_plan" in result
